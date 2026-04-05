import moto
import casadi as cs
import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin

from example_robot_data import load

full = True
soft = False


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
            self.make_foot_kin_constr(i) for i in range(len(foot_frames))
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
            cs.vcat([res, -res]) if soft else res,
            # cs.vcat([v_f[:2, i], z_f[i]]) * active_foot[i],
        )
        c.enable_if_all([self.f_f[i]])

        return c.cast_ineq() if soft else c

    def make_foot_kin_cost(self, i: int):
        pos_cost = (
            moto.cost.create(
                f"kin_pos_cost_{self.foot_frames[i]}", self.pos_args, self.z_f[i]
            )
            .set_gauss_newton(
                moto.sym.params(f"W_kin_{self.foot_frames[i]}", 1, default_val=1e3)
            )
        )
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
        q_nom_res = cpin.difference(self, self.q.sx, self.q_nom.sx)
        state_cost = (
            100.0 * cs.sumsqr(q_nom_res[:3])
            + 100.0 * cs.sumsqr(q_nom_res[3 : 6])
            + 1 * cs.sumsqr(q_nom_res[6 :])
            + 1.0 * cs.sumsqr(self.v_stack[:3])
            + 1.0 * cs.sumsqr(self.v_stack[3 : 6])
            + 0.01 * cs.sumsqr(self.v_stack[6:])
        )
        state_args = self.pos_args + self.vel_args
        cost = moto.cost.create("c_mpc", state_args + [self.q_nom], state_cost)
        if terminal:
            return cost
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
q_d[2] -= 0.02
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
if full:
    prob.add(model.fric)
prob.add(model.kin_constr)
if not full:
    prob.add(model.kin_cost)
# prob.add(model.zf_constr)
prob.add(model.make_tq_limit_constr())
prob.add(model.make_joint_limit_constr())
model.add_dt_constr_and_cost(prob, dt_nom)
prob.add(model.get_state_cost())
prob.add(model.get_input_cost())
# prob.add(model.make_foot_lift_cost(lifted=True))

prob_term = prob.clone()
prob_term.add_terminal(model.get_state_cost(terminal=True))

prob.print_summary()
print("--" * 15)

N_horizon = 50

gait = "trot"
# gait = "hopping"
gait_setting = {
    "trot": [1, 1, 0, 0],
    "hopping": [0, 0, 0, 0],
}
sqp = moto.sqp(n_job=10)
g = sqp.graph
n0 = g.add_head(sqp.create_node(prob))
n1 = g.add_tail(sqp.create_node(prob_term))
g.connect(n0, n1, N_horizon)

sqp.settings.ipm.mu0 = 0.1
sqp.settings.ipm.mu_method = moto.sqp.adaptive_mu_t.mehrotra_predictor_corrector
sqp.settings.ipm_conditional_corrector = True
sqp.settings.prim_tol = 1e-3
sqp.settings.dual_tol = 1e-3
sqp.settings.comp_tol = 1e-3
sqp.settings.rf.max_iters = 2


def stance_ref(data: moto.sqp.data_type):
    global node_idx, current_time
    # periodic base pose reference (smooth oscillation + slow drift between cfg waypoints)
    freq = 1  # Hz
    phase = 2 * np.pi * freq * (current_time + node_idx * model.dt)

    # center and slow drift between the two cfg waypoints over the whole horizon
    center_x, center_y = 0.0, 0.0

    # oscillation amplitudes
    amp_xy = 0.05  # meters lateral/longitudinal
    amp_z = 0.05  # vertical

    # set periodic reference for base x, y
    data.value[model.q_nom][0] = center_x + amp_xy * np.sin(phase)
    data.value[model.q_nom][1] = center_y + amp_xy * np.cos(phase)

    # vertical periodic motion: larger for hopping, small bob for walking/trot
    data.value[model.q_nom][2] = q_d[2] + amp_z * np.sin(2 * phase)

    node_idx += 1


import time
import mujoco
import mujoco.viewer

mj_model = mujoco.MjModel.from_xml_path("example/quadruped/rsc/scene.xml")
mj_data = mujoco.MjData(mj_model)
sim_dt = 0.005
mj_model.opt.timestep = sim_dt
q_d_mujoco = np.copy(q_d)
q_d_mujoco[3:7] = np.array(
    [q_d[6], q_d[3], q_d[4], q_d[5]]
)  # convert to mujoco quaternion format
mj_data.qpos[:] = q_d_mujoco
mujoco.mj_forward(mj_model, mj_data)

cnt = 0
iters = 0
sqp.settings.ipm.warm_start = False

# warm start
node_idx = 0
current_time = 0.0
sqp.apply_forward(stance_ref)
# n0.data.value[model.k_f] = 0
nodes = sqp.graph.flatten_nodes()
data = go2.model.createData()
for n in nodes[:10]:
    for f in model.f_f:
        n.value[model.k_f] = 0
assert n0.addr == nodes[0].addr
sqp.update(100, verbose=True)
start = time.perf_counter()
sqp.settings.ipm.warm_start = True
control_freq = 10  # Hz
update_interval = int(
    1 / (control_freq * model.dt)
)  # number of sim steps between MPC updates

with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    while viewer.is_running():
        loop_start = time.perf_counter()
        # Update initial state for MPC
        n0.data.value[model.q] = mj_data.qpos.copy()
        # convert mujoco quaternion to pinocchio format
        n0.data.value[model.q][3:7] = np.array(
            [mj_data.qpos[4], mj_data.qpos[5], mj_data.qpos[6], mj_data.qpos[3]]
        )
        print("Current base:", n0.data.value[model.q])
        n0.data.value[model.v] = mj_data.qvel.copy()
        print("Current base velocity:", n0.data.value[model.v])

        # Update reference trajectory
        node_idx = 0
        current_time = mj_data.time
        sqp.apply_forward(stance_ref)
        print(f"Updated reference trajectory for node {node_idx}")
        # Run MPC iteration
        mpc_st = time.perf_counter()
        res = sqp.update(5, verbose=False)
        mpc_ed = time.perf_counter()
        print(
            f"MPC iteration {res.num_iter}, prim_res: {res.inf_prim_res:.3e}, dual res: {res.inf_dual_res:.3e}, timing: {(mpc_ed - mpc_st) * 1000:.3f} ms"
        )
        cnt += 1
        iters += res.num_iter

        # Extract and apply control
        # for n in nodes:

        print("n0 reaction forces:", [n0.data.value[f] / dt for f in model.f_f])
        for i, n in enumerate(nodes[0:5]):
            print(f"n{i} base state:\t\t", n.value[model.q][:7])
            print(f"n{i} base velocity:\t", n.value[model.v][:6])
            pin.forwardKinematics(go2.model, data, n.value[model.q], n.value[model.v])
            pin.updateFramePlacements(go2.model, data)
            print("foot heights:", [data.oMf[f].translation[2] for f in model.foot_idx])
            print(
                "foot velocities:",
                [
                    pin.getFrameVelocity(
                        go2.model, data, f, pin.LOCAL_WORLD_ALIGNED
                    ).linear
                    for f in model.foot_idx
                ],
            )
            print("torques:", n.value[model.tq])
            print(
                "reaction forces:",
                [n.value[f] / dt for f in model.f_f],
            )
            # check the forward dynamics acceleration
            # pin.computeJointJacobians(go2.model, data, n.value[model.q])
            # Jac = [
            #     pin.getFrameJacobian(go2.model, data, f, pin.LOCAL_WORLD_ALIGNED)[:3, :]
            #     for f in model.foot_idx
            # ]
            # a = pin.aba(
            #     go2.model,
            #     data,
            #     n.value[model.q],
            #     n.value[model.v],
            #     np.concatenate([np.zeros(6), n.value[model.tq]])
            #     + sum(Jac[i].T @ n.value[f] for i, f in enumerate(model.f_f)) / dt,
            # )
            # print("forward dyn acc:", a)

        step = 0
        while step < update_interval:
            current_node_idx = int(step * sim_dt // dt + 1)
            mj_data.ctrl[:] = nodes[current_node_idx].value[model.tq]
            qj_mujoco = mj_data.qpos[7:]
            qj_mpc = nodes[current_node_idx].value[model.q][7:]
            vj_mujoco = mj_data.qvel[6:]
            vj_mpc = nodes[current_node_idx].value[model.v][6:]
            mj_data.ctrl[:] += 50.0 * (qj_mpc - qj_mujoco) + 0.5 * (vj_mpc - vj_mujoco)
            mujoco.mj_step(mj_model, mj_data)
            step += 1
        viewer.sync()

        while time.perf_counter() - loop_start < update_interval * sim_dt:
            pass  # busy wait to maintain real-time control frequency

print(f"sqp.update() took {(time.perf_counter() - start) / max(1, cnt):.3f} seconds")
print(
    f"per iteration took {(time.perf_counter() - start) / max(1, iters) * 1000:.3f} ms"
)
