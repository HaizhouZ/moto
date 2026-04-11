import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

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
        self.v_f, self.z_f = self.compute_foot_kinematics(self.q_stack, self.v_stack)

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


        self.mu = moto.sym.params("mu", default_val=0.7)  # friction coefficient
        self.fric = [self.make_fric_cone(i, f) for i, f in enumerate(self.f_f)]

        self.q_nom = moto.sym.params(
            "q_nom",
            self.nq,
            default_val=q_nom if q_nom is not None else np.zeros(self.nq),
        )

                # implicit euler
        def implicit_euler():
            q_next = cpin.integrate(self, self.q.sx, (self.v + self.aba) * dt)
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

        self.dyn = self.make_dynamics()


    def make_dynamics(self):
        args = (
            self.pos_args
            + self.vel_args
            + self.pos_args_n
            + self.vel_args_n
            + self.acc_args
            + [*self.f_f]
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

    def compute_foot_kinematics(self, q_stack, v_stack):
        data = self.createData()
        cpin.forwardKinematics(self, data, q_stack, v_stack)
        cpin.computeJointJacobians(self, data)
        cpin.updateFramePlacements(self, data)
        v_f = cs.hcat(
            [
                cpin.getFrameVelocity(self, data, f, pin.LOCAL_WORLD_ALIGNED).linear
                for f in self.foot_idx
            ]
        )
        z_f = cs.vcat([data.oMf[f].translation[2] for f in self.foot_idx])
        return v_f, z_f

    def make_foot_kin_constr(self, i: int, soft: bool = False):
        pos_args = self.pos_args
        vel_args = self.vel_args
        v_f = self.v_f
        z_f = self.z_f
        res = cs.vcat([v_f[:2, i], self.k_f * z_f[i] + v_f[2, i]])
        c = moto.constr.create(
            f"kin_{self.foot_frames[i]}" + ("_soft" if soft else ""),
            pos_args + vel_args + [self.k_f],  # self.z_clip],
            # [self.q, k_f, *active_foot, z_clip],
            # cs.vcat([self.v_f[:2, i], self.k_f * cs.tanh(self.z_f[i]) * self.z_clip + self.v_f[2, i]]) * self.active_foot[i],
            res,  # if not soft else cs.vcat([-res, res]),
            # res,
            # cs.vcat([v_f[:2, i], z_f[i]]) * active_foot[i],
        )
        c.enable_if_all([self.f_f[i]])
        # return c
        if not soft:
            return c
        else:
            soft_c: moto.pmm_constr = c.cast_soft()
            soft_c.rho = 1e-8
        return soft_c

    def make_foot_kin_cost(self, i: int):
        pos_args = self.pos_args
        z_f = self.z_f
        pos_cost = (
            moto.cost.create(
                f"kin_pos_cost_{self.foot_frames[i]}", pos_args, z_f[i]
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
        return moto.ineq.create(
            "q_limit",
            [self.q, self.v],
            cs.vcat([qj, vj]),
            cs.vcat([q_min, -v_lim]),
            cs.vcat([q_max, v_lim]),
        )

    def make_tq_limit_constr(self):
        tq_limit = model.effortLimit[-self.nj :]
        in_arg = [self.tq]
        # in_arg = self.pos_args + self.vel_args + self.acc_args + [*self.active_foot, *self.f_f] + ([self.dt] if isinstance(self.dt, cs.SX) else [])
        return moto.ineq.create(
            "tq_limit", in_arg, self.tq.sx, -tq_limit, tq_limit
        )

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
        c = moto.ineq.create(
            f"fric_{self.foot_frames[i]}", [f, self.mu], cone
        )
        c.enable_if_all([f])
        return c

    def add_dt_constr_and_cost(self, prob: moto.ocp, dt_nom: moto.var):
        if isinstance(self.dt, cs.SX):
            dt_bound = moto.sym.params(
                "dt_bound", 2, default_val=np.array([1e-4, 5e-2])
            )  # bound on dt
            dt_constr = moto.ineq.create("dt", [self.dt, dt_bound], self.dt.sx, dt_bound[0], dt_bound[1])
            # dt_constr = moto.constr("dt_fix", [self.dt], self.dt - 2e-2)
            prob.add(dt_constr)
            W_dt = moto.sym.params("W_dt", 1, default_val=1e8)
            timing_cost = moto.cost.create(
                "c_t", [self.dt, dt_nom, W_dt], W_dt * cs.sumsqr(self.dt - dt_nom)
            )
            prob.add(timing_cost)

    def get_state_cost(self, terminal: bool = False):
        q_stack = self.q_stack
        v_stack = self.v_stack
        q_nom_res = q_stack - self.q_nom
        state_cost = (
            100.0 * cs.sumsqr(q_nom_res[: self.nqb])
            + 1 * cs.sumsqr(q_nom_res[self.nqb :])
            + 1.0 * cs.sumsqr(v_stack[:6])
            + 0.01 * cs.sumsqr(v_stack[6:])
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
display_env = os.getenv("MOTO_DISPLAY")
display = display_env != "0" if display_env is not None else True
if display_env is None and (
    os.getenv("MOTO_SQP_BENCH_RUNS") is not None or
    os.getenv("MOTO_PROFILE_SQP") is not None
):
    display = False
try:
    go2 = load("go2", display=display, verbose=True)
except Exception as exc:
    if display:
        print(f"viewer init failed, retrying with display disabled: {exc}")
        display = False
        go2 = load("go2", display=False, verbose=True)
    else:
        raise
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

def build_stage_node_prob(robot: pinCasadiModel):
    stage_node_prob = moto.node_ocp.create()
    if full:
        stage_node_prob.add(robot.fric)
    stage_node_prob.add(robot.kin_constr)
    if not full:
        stage_node_prob.add(robot.kin_cost)
    robot.add_dt_constr_and_cost(stage_node_prob, dt_nom)
    stage_node_prob.add(robot.make_joint_limit_constr())
    stage_node_prob.add(robot.make_tq_limit_constr())
    stage_node_prob.add(robot.get_state_cost())
    stage_node_prob.add(robot.get_input_cost())
    # stage_node_prob.add(robot.make_foot_lift_cost(lifted=True))
    return stage_node_prob


stage_node_proto = build_stage_node_prob(model)
terminal_node_proto = stage_node_proto.clone()

N_horizon = 100

# setup gait
steps = 4
nodes_per_step = 20
total_gait_steps = steps * nodes_per_step
stance_length = int((N_horizon - total_gait_steps) / 2)
print(f"stance_length: {stance_length}, nodes_per_step: {nodes_per_step}")

gait = "trot"
# gait = "hopping"
gait_setting = {
    "trot": [1, 1, 0, 0],
    "hopping": [0, 0, 0, 0],
}
sqp = moto.sqp(n_job=10)
modeled = sqp.create_graph()


def create_phase_config(step):
    constr_to_disable = []
    for idx, f in enumerate([0, 3, 1, 2]):
        if step % 2 == 0:
            if not gait_setting[gait][idx]:
                constr_to_disable.append(model.f_f[f])
        else:
            if gait_setting[gait][idx]:
                constr_to_disable.append(model.f_f[f])
    return moto.ocp.active_status_config(deactivate_list=constr_to_disable)


segment_lengths = [stance_length]
segment_lengths.extend([nodes_per_step] * steps)
segment_lengths.append(stance_length)


def add_segment(start, end_prob, n_edges):
    end_node = modeled.create_node(end_prob)
    edges = modeled.add_path(start, end_node, n_edges)
    for edge in edges:
        edge.add(model.dyn)
    return end_node


current_node = modeled.create_node(stage_node_proto.clone())
segment_end_nodes = [stage_node_proto.clone(create_phase_config(step)) for step in range(1, steps + 1)]
segment_end_nodes.append(stage_node_proto.clone())
segment_end_nodes.append(terminal_node_proto)
for end_prob, n_edges in zip(segment_end_nodes, segment_lengths):
    current_node = add_segment(current_node, end_prob, n_edges)

if os.getenv("MOTO_DEBUG_SOLVER_PROBS"):
    flat_nodes = sqp.graph.flatten_nodes()
    print("--" * 15)
    print("Terminal node prototype:")
    terminal_node_proto.print_summary()
    print("Head solver node problem:")
    flat_nodes[0].prob.print_summary()
    print("Tail solver node problem:")
    flat_nodes[-1].prob.print_summary()

if os.getenv("MOTO_DEBUG_GRAPH_LAYOUT"):
    print("Flattened solver graph layout:")
    for idx, node in enumerate(sqp.graph.flatten_nodes()):
        prob = node.prob
        print(
            f"  node[{idx}] uid={prob.uid} "
            f"x={prob.dim(moto.field___x)} "
            f"u={prob.dim(moto.field___u)} "
            f"y={prob.dim(moto.field___y)} "
            f"dyn={len(prob.exprs(moto.field___dyn))}"
        )
# g.add_edge(n0, n1, N_horizon)
# print(len(g.flatten_nodes()))
# names = [[] for _ in range(len(g.flatten_nodes()))]
# idx = 0
# def print_node_info(node: moto.sqp.data_type):
#     global names, idx
#     names[idx] = []
#     for c in node.prob.exprs(moto.field___u):
#         names[idx].append(c.name)
#     idx += 1


# sqp.apply_forward(print_node_info)
# print("Node variable names:")
# for i, n in enumerate(g.flatten_nodes()):
#     print(f"Node {i}: {names[i]}")

# exit(0)

sqp.settings.ipm.mu0 = 1.0
# sqp.settings.ipm.mu_method = moto.sqp.adaptive_mu_t.mehrotra_predictor_corrector
sqp.settings.ipm.mu_method = moto.sqp.adaptive_mu_t.monotonic_decrease
# sqp.settings.ipm.mu_method = moto.sqp.adaptive_mu_t.quality_function_based
sqp.settings.ipm_conditional_corrector = True
sqp.settings.prim_tol = 1e-3
sqp.settings.dual_tol = 1e-3
sqp.settings.comp_tol = 1e-3
sqp.settings.rf.max_iters = 2
sqp.settings.ls.update_alpha_dual = False
sqp.settings.restoration.enabled = True
sqp.settings.restoration.max_iter = 10
sqp.settings.restoration.rho_eq = 1e-6
# sqp.settings.scaling.scaling_mode = moto.sqp.scaling_settings.mode_gradient
sqp.settings.ls.primal_gamma = 1e-4
sqp.settings.ls.method = moto.ns_sqp.search_method_filter

max_update_iter = int(os.getenv("MOTO_SQP_MAX_ITER", "2"))
bench_runs = int(os.getenv("MOTO_SQP_BENCH_RUNS", "1"))
bench_show_last = os.getenv("MOTO_SQP_BENCH_SHOW_LAST", "1") != "0"
print(f"SQP update iters: {max_update_iter}")
print(f"SQP benchmark runs: {bench_runs}")

# sqp.settings.ls.backtrack_scheme = moto.ns_sqp.backtrack_scheme_geometric
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
# simple hopping reduced
# cfg = [
#     [-0.19444444444444445, -0.17929292929292928],
#     [-0.3888888888888889, -0.35858585858585856],
# ]
step = 0
node_idx = 0

print("")
def gait_setup(data: moto.sqp.data_type):
    global step, node_idx
    ref_node_idx = min(node_idx + 1, N_horizon)
    switch_step = False
    if ref_node_idx >= stance_length and ref_node_idx + stance_length < N_horizon:
        switch_step = (ref_node_idx - stance_length) % nodes_per_step == 0
        for idx, f in enumerate([0, 3, 1, 2]):
            if step % 2 == 0:
                if gait == "hopping":
                    data.value[model.q_nom][2] = 0.5
            else:
                ...
    else:
        ...
    ref_step = step + 1 if switch_step else step
    node_progress = ref_node_idx / N_horizon
    if ref_step >= 1 and ref_step <= 2:
        data.value[model.q_nom][0] = node_progress * cfg[0][0]
        data.value[model.q_nom][1] = node_progress * cfg[0][1]
    elif ref_step > 2:
        data.value[model.q_nom][0] = cfg[0][0] + node_progress * (
            cfg[1][0] - cfg[0][0]
        )
        data.value[model.q_nom][1] = cfg[0][1] + node_progress * (
            cfg[1][1] - cfg[0][1]
        )
    # data.value[model.q] = data.value[model.qn] = data.value[model.q_nom]

    if switch_step:
        step += 1
    node_idx += 1


sqp.apply_forward(gait_setup)
# exit(0)
import time

cnt = 0
iters = 0
start = time.perf_counter()
res = None
for i in range(bench_runs):
    bench_verbose = os.getenv("MOTO_SQP_BENCH_VERBOSE") is not None or (
        bench_show_last and i + 1 == bench_runs
    )
    res = sqp.update(max_update_iter, verbose=bench_verbose)
    sqp.settings.ipm.warm_start = True
    cnt += 1
    iters += res.num_iter
elapsed = time.perf_counter() - start

print(f"sqp.update() took {elapsed / cnt:.3f} seconds")
print(f"per iteration took {elapsed / iters * 1000:.3f} ms")

if os.getenv("MOTO_PROFILE_SQP"):
    report = sqp.get_profile_report()
    print(
        f"profile total={report.total_ms:.1f} ms "
        f"init={report.initialize_ms:.1f} ms "
        f"iters={report.sqp_iterations} "
        f"trial_evals={report.trial_evaluations}"
    )
    top_phases = sorted(report.phases, key=lambda p: p.total_ms, reverse=True)[:10]
    print("top profile phases:")
    for phase in top_phases:
        print(
            f"  {phase.name:24s} total={phase.total_ms:9.3f} ms "
            f"avg={phase.avg_ms:8.3f} ms "
            f"calls={phase.calls:4d} "
            f"share={phase.share_of_update * 100:6.2f}%"
        )
    print("per-iteration profile:")
    for it in report.iterations:
        print(
            f"  iter {it.index:2d}: total={it.total_ms:9.3f} ms "
            f"ls_steps={it.ls_steps:3d} "
            f"trial_evals={it.trial_evaluations:3d}"
        )

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
