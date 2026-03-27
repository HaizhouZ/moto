import moto
import casadi as cs
import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin

from example_robot_data import load
import meshcat.geometry as mg
import meshcat.transformations as tf
import meshcat_shapes as mcs

full = True
soft = True


class pinCasadiModel(cpin.Model):
    def __init__(
        self,
        model: pin.Model,
        name: str = "",
        dt: cs.SX | float = 0.01,
        q_nom: np.ndarray | None = None,
        dense: bool = True,
        foot_frames: list[str] | None = None,
        use_fwd_dyn: bool = False,
    ):
        self.fmodel = model
        super().__init__(model)
        if name:
            model.name = name
        else:
            name = model.name.replace("_description", "")
            self.name = name
        self.is_floating_based: bool = self.nv > self.njoints
        self.nj = self.nv - 6 if self.is_floating_based else self.nv
        self.data = self.createData()
        self.dt = dt
        self.dense = dense
        self.foot_idx = [model.getFrameId(f) for f in foot_frames]
        self.foot_frames = foot_frames
        self.use_fwd_dyn = use_fwd_dyn

        if self.is_floating_based:
            self.nqb = 6
        if self.nq - self.nv == 1:
            self.nqb = 7

        # make_primal
        self.q, self.qn = moto.sym.states(name + "_q", self.nq)
        if q_nom is not None:
            assert q_nom.shape == (self.nq,), "q_nom has wrong shape"
            self.q.default_value = q_nom
            self.qn.default_value = q_nom
        self.v, self.vn = moto.sym.states(name + "_v", self.nv)
        # self.aj = moto.sym.inputs(name + "_aj", self.nj)
        # self.a = moto.sym.inputs(name + "_a", self.nv)
        # self.aj = self.a[-self.nj :]
        # self.ab = (self.vn - self.v)[:6] / dt
        self.a = (self.vn - self.v) / dt
        # self.a = cs.vcat([self.ab, self.aj]) if self.is_floating_based else self.aj
        self.tq = moto.sym.inputs(name + "_tq", self.nj)

        # implicit euler
        def implicit_euler():
            q_next = cpin.integrate(self, self.q.sx, self.vn * dt)
            # v_next = self.v[-self.nj :] + self.aj * dt
            # v_next = self.v + self.a * dt
            # return moto.dense_dynamics(
            #     name + "_euler",
            #     [self.q, self.v, self.qn, self.vn, self.a, dt],
            #     cs.vcat([self.qn - q_next, self.vn - v_next]),
            # )
            # return cs.vcat([self.qn - q_next, self.vn[-self.nj :] - v_next])
            # return cs.vcat([self.qn - q_next, self.vn - v_next])
            return cs.vcat([self.qn - q_next])

        self.joint_euler = implicit_euler()
        # self.ntq = model.nv - 6 if self.is_floating_based else model.nv
        self.q_stack = self.q.sx
        self.v_stack = self.v.sx
        self.qn_stack = self.qn.sx
        self.vn_stack = self.vn.sx
        self.a_stack = self.a
        self.pos_args = [self.q]
        self.vel_args = [self.v]
        # self.acc_args = [self.aj]
        # self.acc_args = [self.a]
        self.acc_args = [self.tq]
        self.pos_args_n = [self.qn]
        self.vel_args_n = [self.vn]

        cpin.forwardKinematics(self, self.data, self.q_stack, self.v_stack)
        cpin.computeJointJacobians(self, self.data)
        cpin.updateFramePlacements(self, self.data)

        self.foot_jacs = [
            cpin.getFrameJacobian(self, self.data, f, pin.LOCAL_WORLD_ALIGNED)[:3, :]
            for f in self.foot_idx
        ]
        # self.active_foot = [moto.sym.params(f"af_{f}", 1, default_val=1) for f in foot_frames]  # active foot indicator
        self.k_f = moto.sym.params("k_f", default_val=100 if full else 0)
        # self.z_clip = moto.sym.params("z_c", 1, default_val=0.05)  # minimum height for the foot to be considered in contact
        self.f_f = [
            moto.sym.inputs(f"f_{f}", 3, default_val=np.array([0.0, 0.0, 0.5]))
            for f in foot_frames
        ]
        self.F_f = [self.foot_jacs[i].T @ self.f_f[i] for i in range(len(foot_frames))]
        self.z_f = cs.vcat([self.data.oMf[f].translation[2] for f in self.foot_idx])

        self.kin_constr = [
            self.make_foot_kin_constr(i, soft) for i in range(len(foot_frames))
        ]
        self.kin_cost = [self.make_foot_kin_cost(i) for i in range(len(foot_frames))]

        # self.zf_constr = [self.make_zero_force_constr(i, f) for i, f in enumerate(self.f_f)]
        if self.use_fwd_dyn:
            tau = (
                cs.vcat([cs.SX.zeros(6), self.tq])
                if self.is_floating_based
                else self.tq
            )
            self.aba = (
                cpin.aba(
                    self,
                    self.data,
                    self.q_stack,
                    self.v_stack,
                    tau + sum(self.F_f) / dt,
                )
                * dt
            )
        else:
            self.rnea = cpin.rnea(
                self, self.data, self.q_stack, self.v_stack, self.a_stack
            ) * dt - sum(self.F_f)
        # if self.is_floating_based:
        #     self.rnea_base = self.rnea[:6]
        # self.tq = self.rnea[-self.nj :]

        self.dyn = self.make_dynamics()

        self.mu = moto.sym.params("mu", default_val=0.7)  # friction coefficient
        self.fric = [self.make_fric_cone(i, f) for i, f in enumerate(self.f_f)]

        self.q_nom = moto.sym.params(
            "q_nom",
            self.nq,
            default_val=q_nom if q_nom is not None else np.zeros(self.nq),
        )

    def make_dynamics(self):
        args = (
            self.pos_args
            + self.vel_args
            + self.pos_args_n
            + self.vel_args_n
            + [*self.f_f]
            + self.acc_args
        )
        if isinstance(self.dt, cs.SX):
            args.append(self.dt)
        out = [self.joint_euler]
        # out = [self.qn - cpin.integrate(self, self.q, self.vn * self.dt), self.vn - self.v - self.aba * self.dt]

        # if self.is_floating_based:
        # out.append(self.rnea_base)
        # return moto.dense_dynamics(self.name + "_fb_id", args, cs.vcat(out))
        in_arg = self.pos_args + self.vel_args + self.acc_args + [*self.f_f]
        # return [moto.dense_dynamics(self.name + "_euler", args, cs.vcat(out)),
        #         moto.constr(self.name + "_id", in_arg, self.rnea[:6])]
        if self.use_fwd_dyn:
            v_next = self.v + self.aba
            return moto.dense_dynamics.create(
                self.name + "_fd", args, cs.vcat(out + [self.vn - v_next])
            )
        else:
            tau = (
                cs.vcat([cs.SX.zeros(6), self.tq])
                if self.is_floating_based
                else self.tq
            )
            return moto.dense_dynamics.create(
                self.name + "_id", args, cs.vcat(out + [self.rnea - tau * self.dt])
            )
        # return moto.dense_dynamics(self.name + "_fb_fd", args, cs.vcat(out))

    def make_foot_kin_constr(self, i: int, soft: bool = False):
        self.v_f = cs.hcat(
            [
                cpin.getFrameVelocity(
                    self, self.data, f, pin.LOCAL_WORLD_ALIGNED
                ).linear
                for f in self.foot_idx
            ]
        )
        res = cs.vcat([self.v_f[:2, i], self.k_f * self.z_f[i] + self.v_f[2, i]])
        c = moto.constr.create(
            f"kin_{self.foot_frames[i]}" + ("_soft" if soft else ""),
            self.pos_args + self.vel_args + [self.k_f],  # self.z_clip],
            # [self.q, k_f, *active_foot, z_clip],
            # cs.vcat([self.v_f[:2, i], self.k_f * cs.tanh(self.z_f[i]) * self.z_clip + self.v_f[2, i]]) * self.active_foot[i],
            res,
            # cs.vcat([v_f[:2, i], z_f[i]]) * active_foot[i],
        )
        c.enable_if_all([self.f_f[i]])
        # return c.cast_soft() if soft else c
        return c
        # soft_c: moto.pmm_constr = c.cast_soft()
        # soft_c.rho = 1e-8
        # return soft_c if soft else c

    def make_foot_kin_cost(self, i: int):
        pos_cost = (
            moto.cost.create(
                f"kin_pos_cost_{self.foot_frames[i]}", self.pos_args, self.z_f[i]
            )
            .set_gauss_newton(
                moto.sym.params(f"W_kin_{self.foot_frames[i]}", 1, default_val=1e3)
            )
            .as_terminal()
        )
        pos_cost.name = pos_cost.name.replace("_terminal", "")
        pos_cost.enable_if_all([self.f_f[i]])
        return pos_cost

    def make_joint_limit_constr(self):
        q_min = model.lowerPositionLimit[-self.nj :]
        q_max = model.upperPositionLimit[-self.nj :]
        v_lim = model.velocityLimit[-self.nj :]
        print("Joint limits:")
        print("q_min:", q_min)
        print("q_max:", q_max)
        print("v_lim:", v_lim)
        qj = self.q[-self.nj :]
        vj = self.v[-self.nj :]
        return moto.constr.create(
            "q_limit",
            [self.q, self.v],
            cs.vcat([q_min - qj, qj - q_max, vj - v_lim, -vj - v_lim]),
        ).cast_ineq()

    def make_tq_limit_constr(self):
        tq_limit = model.effortLimit[-self.nj :]
        in_arg = [self.tq]
        # in_arg = self.pos_args + self.vel_args + self.acc_args + [*self.active_foot, *self.f_f] + ([self.dt] if isinstance(self.dt, cs.SX) else [])
        return moto.constr.create(
            "tq_limit", in_arg, cs.vcat([self.tq - tq_limit, -self.tq - tq_limit])
        ).cast_ineq()

    def make_fric_cone(self, i, f: moto.var):
        cone = cs.vcat(
            [
                f[0] - self.mu * f[2],
                -f[0] - self.mu * f[2],
                f[1] - self.mu * f[2],
                -f[1] - self.mu * f[2],
            ]
        )
        # cone = f[0] - self.mu * cs.sqrt(cs.sumsqr(f[1:]) + 1e-9)
        c = moto.constr.create(
            f"fric_{self.foot_frames[i]}", [f, self.mu], cone
        ).cast_ineq()
        c.enable_if_all([f])
        return c

    def add_dt_constr_and_cost(self, prob: moto.ocp, dt_nom: moto.var):
        if isinstance(self.dt, cs.SX):
            dt_bound = moto.sym.params(
                "dt_bound", 2, default_val=np.array([1e-4, 5e-2])
            )  # bound on dt
            dt_constr = moto.constr.create(
                "dt",
                [self.dt, dt_bound],
                cs.vcat([dt_bound[0] - self.dt, self.dt - dt_bound[1]]),
            ).cast_ineq()
            # dt_constr = moto.constr("dt_fix", [self.dt], self.dt - 2e-2)
            prob.add(dt_constr)
            W_dt = moto.sym.params("W_dt", 1, default_val=1e8)
            timing_cost = moto.cost.create(
                "c_t", [self.dt, dt_nom, W_dt], W_dt * cs.sumsqr(self.dt - dt_nom)
            )
            prob.add(timing_cost)

    def get_state_cost(self, terminal: bool = False):
        q_nom_res = self.q_stack - self.q_nom
        state_cost = (
            100.0 * cs.sumsqr(q_nom_res[: self.nqb])
            + 1 * cs.sumsqr(q_nom_res[self.nqb :])
            + 1.0 * cs.sumsqr(self.v_stack[:6])
            + 0.01 * cs.sumsqr(self.v_stack[6:])
        )
        state_args = self.pos_args + self.vel_args
        cost = moto.cost.create("c", state_args + [self.q_nom], state_cost)
        if terminal:
            return cost.as_terminal()
        return cost

    def get_input_cost(self):
        input_args = self.acc_args + [*self.f_f]
        input_cost = 1e-6 * cs.sumsqr(self.tq) + 1e-3 * cs.sumsqr(cs.vcat(self.f_f))
        return moto.cost.create("c_u", input_args, input_cost)

    def make_foot_lift_cost(self, lifted: bool = True):
        self.z_f_lift_d = moto.sym.params(
            "z_f_lift_d", 1, default_val=0.07
        )  # desired foot lift height
        if lifted:
            self.z_f_d = moto.sym.inputs(
                "z_f_d", 4, default_val=0.0
            )  # desired foot height when in contact
            foot_lift_constr = moto.constr.create(
                "foot_lift_constr",
                self.pos_args + [self.z_f_d],
                (self.z_f - self.z_f_d),
            )
            foot_lift_cost = moto.cost.create(
                "c_z",
                [self.z_f_d, self.z_f_lift_d],
                100 * cs.sumsqr((self.z_f_d - self.z_f_lift_d)),
            )
            foot_lift_constr.disable_if([*self.f_f])
            foot_lift_cost.disable_if([*self.f_f])
            return [foot_lift_constr, foot_lift_cost]
        else:
            foot_lift_cost = moto.cost(
                "c_z",
                self.pos_args + [self.z_f_lift_d, *self.active_foot],
                100
                * cs.sumsqr(
                    (self.z_f - self.z_f_lift_d) * (1 - cs.vcat(self.active_foot))
                ),
            ).set_gauss_newton()
            return foot_lift_cost


# dt = moto.sym.inputs("dt", 1, default_val=0.02)
dt_nom = moto.sym.params("dt_nom", 1, default_val=0.02)
dt = 0.02
display = True
go2 = load("go2", display=display, verbose=True)
q_d = np.copy(go2.q0)
root_joint = pin.JointModelComposite()
root_joint.addJoint(pin.JointModelTranslation())
root_joint.addJoint(pin.JointModelSpherical())
# root_joint.addJoint(pin.JointModelSphericalZYX())
# q_d = np.concatenate((q_d[:3], np.array([0.0, 0.0, 0.0]), q_d[7:]))
model = pin.buildModelFromUrdf(go2.urdf, root_joint)
np.set_printoptions(precision=3, suppress=True, linewidth=200)
foot_frames = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
model = pinCasadiModel(
    model, dt=dt, q_nom=q_d, dense=True, foot_frames=foot_frames, use_fwd_dyn=True
)

prob = moto.ocp.create()
prob.add(model.dyn)
# if full:
#     prob.add(model.fric)
prob.add(model.kin_constr)
if not full:
    prob.add(model.kin_cost)
# prob.add(model.zf_constr)
# prob.add(model.make_tq_limit_constr())
# prob.add(model.make_joint_limit_constr())
model.add_dt_constr_and_cost(prob, dt_nom)
prob.add(model.get_state_cost())
prob.add(model.get_input_cost())
# prob.add(model.make_foot_lift_cost(lifted=True))

prob_term = prob.clone()
prob_term.add(model.get_state_cost(terminal=True))

prob.print_summary()
print("--" * 15)

N_horizon = 100

# setup gait
steps = 4
nodes_per_step = 20
total_gait_steps = steps * nodes_per_step
stance_length = int((N_horizon - total_gait_steps) / 2)
print(f"stance_length: {stance_length}, nodes_per_step: {nodes_per_step}")

# gait = "trot"
gait = "hopping"
gait_setting = {
    "trot": [1, 1, 0, 0],
    "hopping": [0, 0, 0, 0],
}
sqp = moto.sqp(n_job=10)
g = sqp.graph
n0 = g.set_head(g.add(sqp.create_node(prob)))


def create_phase_problem(step):
    constr_to_disable = []
    for idx, f in enumerate([0, 3, 1, 2]):
        if step % 2 == 0:
            if not gait_setting[gait][idx]:
                constr_to_disable += [model.f_f[f]]
        else:
            if gait_setting[gait][idx]:
                constr_to_disable += [model.f_f[f]]
    phase_prob = prob.clone(
        moto.ocp.active_status_config(deactivate_list=constr_to_disable)
    )
    return phase_prob


for step in range(steps):
    np = g.add(sqp.create_node(create_phase_problem(step + 1)))
    np.data.prob.print_summary()
    if step == 0:
        g.add_edge(n0, np, stance_length, include_ed=False)
    else:
        g.add_edge(n_prev, np, nodes_per_step, include_ed=False)
    n_prev = np

nstop = g.add(sqp.create_node(prob))
g.add_edge(n_prev, nstop, nodes_per_step, include_ed=False)
n1 = g.set_tail(g.add(sqp.create_node(prob_term)))
g.add_edge(nstop, n1, stance_length)
# g.add_edge(n0, n1, N_horizon)
# print(g.flatten_nodes())

sqp.settings.ipm.mu0 = 1.0
sqp.settings.ipm.mu_method = moto.sqp.adaptive_mu_t.mehrotra_predictor_corrector
# sqp.settings.ipm.mu_method = moto.sqp.adaptive_mu_t.monotonic_decrease
# sqp.settings.ipm.mu_method = moto.sqp.adaptive_mu_t.quality_function_based
sqp.settings.ipm_conditional_corrector = True
sqp.settings.prim_tol = 1e-3
sqp.settings.dual_tol = 1e-3
sqp.settings.comp_tol = 1e-3
sqp.settings.rf.max_iters = 2
sqp.settings.ls.max_soc_iter = 4
sqp.settings.ls.update_alpha_dual = True
sqp.settings.scaling.scaling_mode = moto.sqp.scaling_settings.mode_none
# cfg = [
#     [-0.9595959595959596, -0.6161616161616161],
#     [-0.9595959595959596, 0.31313131313131315],
# ]
# trot reduced hard
cfg = [
    [-0.8383838383838383, 0.2525252525252526],
    [-0.4949494949494949, 0.8989898989898992],
]
cfg = [
    [-0.8181818181818181, 0.3737373737373739],
    [0.4545454545454546, -0.21212121212121204],
]
step = 0
node_idx = 0


def gait_setup(data: moto.sqp.data_type):
    global step, node_idx
    if node_idx >= stance_length and node_idx + stance_length < N_horizon:
        switch_step = (node_idx - stance_length) % nodes_per_step == 0
        if switch_step:
            step += 1
        for idx, f in enumerate([0, 3, 1, 2]):
            if step % 2 == 0:
                if gait == "hopping":
                    data.value[model.q_nom][2] = 0.5
            else:
                ...
    else:
        ...
    if step >= 1 and step <= 2:
        data.value[model.q_nom][0] = node_idx / N_horizon * cfg[0][0]
        data.value[model.q_nom][1] = node_idx / N_horizon * cfg[0][1]
    elif step > 2:
        data.value[model.q_nom][0] = cfg[0][0] + node_idx / N_horizon * (
            cfg[1][0] - cfg[0][0]
        )
        data.value[model.q_nom][1] = cfg[0][1] + node_idx / N_horizon * (
            cfg[1][1] - cfg[0][1]
        )
    # data.value[model.q] = data.value[model.qn] = data.value[model.q_nom]

    node_idx += 1


sqp.apply_forward(gait_setup)
import time

cnt = 0
iters = 0
start = time.perf_counter()
# while cnt < 50:
res = sqp.update(50, verbose=True)
sqp.settings.ipm.warm_start = True
cnt += 1
iters += res.num_iter
print(f"sqp.update() took {(time.perf_counter() - start) / cnt:.3f} seconds")
print(f"per iteration took {(time.perf_counter() - start) / iters * 1000:.3f} ms")

q_res = []
dt_res = []
node_idx = 0


def get_sym(node: moto.sqp.data_type):
    global node_idx
    q_res.append(node.value[model.q])
    if isinstance(dt, float):
        dt_res.append(dt)
    else:
        dt_res.append(node.value[dt])
    node_idx += 1
    if node_idx >= N_horizon:
        q_res.append(node.value[model.qn])


sqp.apply_forward(get_sym)
exit(0)
if display:
    import time

    viz = go2.viz

    target = mg.Sphere(0.02)

    color = [0x00FF00, 0xFF0000]
    for i in range(2):
        target = mg.Sphere(0.02)
        mcs.point(viz.viewer[f"/target{i}"], color=color[i], radius=0.04)
        pose = tf.compose_matrix(translate=cfg[i][:2] + [0.3])
        viz.viewer[f"/target{i}"].set_transform(pose)
        mcs.frame(viz.viewer[f"/frame{i}"])
        viz.viewer[f"/frame{i}"].set_transform(pose)

    while True:
        for i in range(len(q_res)):
            start = time.perf_counter()
            go2.display(q_res[i])
            if i is not N_horizon:
                dt_ = dt_res[i]
                while time.perf_counter() - start < dt_:
                    pass
        time.sleep(0.5)

# def print_sym(node: moto.sqp.data_type):
#     node.sym.print()
#     node.print_residuals()

# sqp.apply_forward(print_sym, early_stop=20)
