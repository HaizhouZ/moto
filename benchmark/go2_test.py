import moto
import casadi as cs
import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin

from example_robot_data import load
from benchmark_tool import benchmark_tool
from example.helpers import pinocchio_states


benchmark = benchmark_tool()
benchmark.arg_parser.add_argument("--soft", action="store_true")
benchmark.arg_parser.add_argument("--full", action="store_true")


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
        self.q, self.qn, self.v, self.vn = pinocchio_states(self, name, q_nom)
        self.a = (self.vn - self.v) / dt
        self.tq = moto.sym.inputs(name + "_tq", self.nj)

        def implicit_euler():
            q_next = self.q.symbolic_integrate(self.q.sx, self.vn * dt)
            return cs.vcat([self.q.symbolic_difference(self.qn.sx, q_next)])

        self.joint_euler = implicit_euler()
        # self.ntq = model.nv - 6 if self.is_floating_based else model.nv
        self.q_stack = self.q.sx
        self.v_stack = self.v.sx
        self.qn_stack = self.qn.sx
        self.vn_stack = self.vn.sx
        self.a_stack = self.a
        cpin.forwardKinematics(self, self.data, self.q_stack, self.v_stack)
        cpin.computeJointJacobians(self, self.data)
        cpin.updateFramePlacements(self, self.data)

        self.foot_jacs = [
            cpin.getFrameJacobian(self, self.data, f, pin.LOCAL_WORLD_ALIGNED)[:3, :]
            for f in self.foot_idx
        ]
        # self.active_foot = [moto.sym.params(f"af_{f}", 1, default_val=1) for f in foot_frames]  # active foot indicator
        self.k_f = moto.sym.params(
            "k_f", default_val=100 if benchmark.args.full else 0
        )  # kinematics constraint gain
        self.f_f = [
            moto.sym.inputs(f"f_{f}", 3, default_val=np.array([0.0, 0.0, 0.5]))
            for f in foot_frames
        ]
        self.F_f = [self.foot_jacs[i].T @ self.f_f[i] for i in range(len(foot_frames))]
        self.z_f: cs.SX = cs.vcat(
            [self.data.oMf[f].translation[2] for f in self.foot_idx]
        )

        self.kin_constr = [
            self.make_foot_kin_constr(i, soft=benchmark.args.soft)
            for i in range(len(foot_frames))
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
        out = [self.joint_euler]
        if self.use_fwd_dyn:
            v_next = self.v + self.aba
            return moto.dense_dynamics.create(
                self.name + "_fd", cs.vcat(out + [self.vn - v_next])
            )
        else:
            tau = (
                cs.vcat([cs.SX.zeros(6), self.tq])
                if self.is_floating_based
                else self.tq
            )
            return moto.dense_dynamics.create(
                self.name + "_id", cs.vcat(out + [self.rnea - tau * self.dt])
            )

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
            cs.vcat([res, -res]) if soft else res,
        )
        c.enable_if_all([self.f_f[i]])

        return c.cast_ineq() if soft else c

    def make_foot_kin_cost(self, i: int):
        pos_cost = moto.cost.from_scalar(
            f"kin_pos_cost_{self.foot_frames[i]}", self.z_f[i],
            weight=1e3,
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
        return moto.ineq.create(
            "q_limit",
            cs.vcat([qj, vj]),
            np.concatenate([q_min, -v_lim]),
            np.concatenate([q_max, v_lim]),
        )

    def make_tq_limit_constr(self):
        tq_limit = model.effortLimit[-self.nj :]
        return moto.ineq.bounds("tq_limit", self.tq, -tq_limit, tq_limit)

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
        c = moto.constr.create(f"fric_{self.foot_frames[i]}", cone).cast_ineq()
        return c

    def add_dt_constr_and_cost(self, prob: moto.stage_ocp, dt_nom: moto.var):
        if isinstance(self.dt, cs.SX):
            dt_bound = moto.sym.params(
                "dt_bound", 2, default_val=np.array([1e-4, 5e-2])
            )  # bound on dt
            dt_constr = moto.constr.create(
                "dt",
                cs.vcat([dt_bound[0] - self.dt, self.dt - dt_bound[1]]),
            ).cast_ineq()
            # dt_constr = moto.constr("dt_fix", [self.dt], self.dt - 2e-2)
            prob.add(dt_constr)
            timing_cost = moto.cost.from_scalar(
                "c_t", self.dt - dt_nom, weight=2e8
            )
            prob.add(timing_cost)

    def get_state_cost(self):
        q_nom_res = self.q.symbolic_difference(self.q.sx, self.q_nom.sx)
        residual = cs.vcat([q_nom_res, self.v_stack])
        weight = np.r_[
            np.full(self.nqb, 200.0),
            np.full(q_nom_res.numel() - self.nqb, 2.0),
            np.full(6, 2.0),
            np.full(self.v_stack.numel() - 6, 0.02),
        ]
        cost = moto.cost.from_vector("c", residual, weight=weight)
        return cost

    def get_input_cost(self):
        forces = cs.vcat(self.f_f)
        return moto.cost.from_vector(
            "c_u", cs.vcat([self.tq, forces]),
            weight=np.r_[np.full(self.tq.numel(), 2e-6),
                         np.full(forces.numel(), 2e-3)],
        )

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
                (self.z_f - self.z_f_d),
            )
            foot_lift_cost = moto.cost.from_vector(
                "c_z", self.z_f_d - self.z_f_lift_d, weight=200.0,
            )
            foot_lift_constr.disable_if([*self.f_f])
            foot_lift_cost.disable_if([*self.f_f])
            return [foot_lift_constr, foot_lift_cost]
        else:
            foot_lift_cost = moto.cost.from_vector(
                "c_z",
                (self.z_f - self.z_f_lift_d)
                * (1 - cs.vcat(self.active_foot)), weight=200.0,
            )
            return foot_lift_cost


# dt = moto.sym.inputs("dt", 1, default_val=0.02)
dt_nom = moto.sym.params("dt_nom", 1, default_val=0.02)
dt = 0.02
display = False
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
model.joint_limit_constr = model.make_joint_limit_constr()
model.state_cost = model.get_state_cost()

prob = moto.stage_ocp.create()
prob.add(model.dyn)
if benchmark.args.full:
    prob.add(model.fric)
prob.st.add(model.kin_constr)
if not benchmark.args.full:
    prob.st.add(model.kin_cost)
# prob.add(model.zf_constr)
prob.add(model.make_tq_limit_constr())
prob.st.add(model.joint_limit_constr)
model.add_dt_constr_and_cost(prob, dt_nom)
prob.st.add(model.state_cost)
prob.add(model.get_input_cost())
# prob.add(model.make_foot_lift_cost(lifted=True))

prob.print_summary()
print("--" * 15)

N_horizon = 100


gaits = ["trot", "hopping"]
gait_setting = {
    "trot": [1, 1, 0, 0],
    "hopping": [0, 0, 0, 0],
}

from tqdm import tqdm
import itertools

for gait, (idx_cfg, cfg) in tqdm(
    itertools.product(gaits, enumerate(benchmark.config)),
    total=len(gaits) * len(benchmark.config),
):
    assert isinstance(cfg, list) and len(cfg) == 2
    sqp = moto.sqp(n_job=10)
    # setup gait
    steps = 4
    nodes_per_step = 20
    total_gait_steps = steps * nodes_per_step
    stance_length = int((N_horizon - total_gait_steps) / 2)

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
            moto.active_status_config(deactivate_list=constr_to_disable)
        )
        return phase_prob

    phase_lengths = [stance_length]
    phase_lengths.extend([nodes_per_step] * steps)
    phase_lengths.append(stance_length)

    phase_prototypes = [prob]
    phase_prototypes.extend(create_phase_problem(step) for step in range(1, steps + 1))
    phase_prototypes.append(prob.clone())
    graph_stages = []
    for prototype, n_edges in zip(phase_prototypes, phase_lengths):
        phase = [prototype.copy() for _ in range(n_edges)]
        sqp.stages.extend(phase)
        graph_stages.extend(phase)
    graph_stages[-1].ed.add(model.kin_constr)
    if not benchmark.args.full:
        graph_stages[-1].ed.add(model.kin_cost)
    graph_stages[-1].ed.add(model.joint_limit_constr)
    graph_stages[-1].ed.add(model.state_cost)

    sqp.settings.ipm.mu0 = 1
    # sqp.settings.mu_method = moto.sqp.adaptive_mu_t.mehrotra_probing
    sqp.settings.ipm.mu_method = moto.sqp.adaptive_mu_t.mehrotra_predictor_corrector
    sqp.settings.ipm_conditional_corrector = True
    sqp.settings.prim_tol = 1e-3
    sqp.settings.dual_tol = 1e-3
    sqp.settings.comp_tol = 1e-3
    sqp.settings.rf.max_iters = 2

    step = 0
    node_idx = 0

    def gait_setup(data: moto.sqp.data_type):
        global step, node_idx, gait
        if node_idx >= stance_length and node_idx + stance_length < N_horizon:
            switch_step = (node_idx - stance_length) % nodes_per_step == 0
            if switch_step:
                step += 1
            for idx, f in enumerate([0, 3, 1, 2]):
                if step % 2 == 0:
                    if gait == "hopping":
                        data.value[model.q_nom][2] = 0.5
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
        # data.value[model.q_nom][0] = node_idx / N_horizon * 2.0
        node_idx += 1

    for node in sqp.nodes:
        gait_setup(node)

    benchmark.run(sqp, gait, idx_cfg, cfg)

    del sqp

benchmark.dump()
