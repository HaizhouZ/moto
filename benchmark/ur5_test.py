import moto
import casadi as cs
import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin

from example_robot_data import load
from benchmark_tool import benchmark_tool

import argparse, json


benchmark = benchmark_tool()

benchmark.arg_parser.add_argument("--soft", action="store_true")
benchmark.arg_parser.add_argument("--cost", action="store_true")

args = benchmark.args


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
        self.use_fwd_dyn = use_fwd_dyn
        self.ee_frame = "ee_link"
        self.ee_id = model.getFrameId(self.ee_frame)

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
        cpin.updateFramePlacements(self, self.data)

        if self.use_fwd_dyn:
            tau = (
                cs.vcat([cs.SX.zeros(6), self.tq])
                if self.is_floating_based
                else self.tq.sx
            )
            self.aba = cpin.aba(self, self.data, self.q_stack, self.v_stack, tau) * dt
        else:
            self.rnea = (
                cpin.rnea(self, self.data, self.q_stack, self.v_stack, self.a_stack)
                * dt
            )

        self.dyn = self.make_dynamics()

        self.q_nom = moto.sym.params(
            "q_nom",
            self.nq,
            default_val=q_nom if q_nom is not None else np.zeros(self.nq),
        )
        q_min = model.lowerPositionLimit[-self.nj :]
        q_max = model.upperPositionLimit[-self.nj :]
        v_lim = model.velocityLimit[-self.nj :]
        self.q_min = q_min
        self.q_max = q_max
        self.v_lim = v_lim

    def make_dynamics(self):
        args = (
            self.pos_args
            + self.vel_args
            + self.pos_args_n
            + self.vel_args_n
            + self.acc_args
        )
        if isinstance(self.dt, cs.SX):
            args.append(self.dt)
        out = [self.joint_euler]
        # out = [self.qn - cpin.integrate(self, self.q, self.vn * self.dt), self.vn - self.v - self.aba * self.dt]

        # if self.is_floating_based:
        # out.append(self.rnea_base)
        # return moto.dense_dynamics(self.name + "_fb_id", args, cs.vcat(out))
        in_arg = self.pos_args + self.vel_args + self.acc_args
        # return [moto.dense_dynamics(self.name + "_euler", args, cs.vcat(out)),
        #         moto.constr.create(self.name + "_id", in_arg, self.rnea[:6])]
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

    def make_ee_pos_constr(self, soft: bool = False, cost: bool = False):
        self.r_des = moto.sym.params("r_des", 3, default_val=np.zeros(3))
        self.quat_des = moto.sym.params(
            "quat_des", 4, default_val=np.array([0.0, 0.0, 0.0, 1.0])
        )
        ee_des = cpin.XYZQUATToSE3(cs.vcat([self.r_des, self.quat_des]))
        ee_pos = self.data.oMf[self.ee_id]
        res = cpin.log6(ee_des.inverse() * ee_pos).np
        if cost:
            if not hasattr(self, "W_ee_cost"):
                self.W_ee_cost = moto.sym.params(
                    "W_ee_cost",
                    6,
                    default_val=np.array([40.0, 40.0, 40.0, 0.0, 0.0, 0.0]),
                )
            return (
                moto.cost.create(
                    "ee_cost", self.pos_args + [self.r_des, self.quat_des], res
                )
                .set_gauss_newton(self.W_ee_cost)
            )
        if not soft:
            return moto.constr.create(
                "ee_constr", self.pos_args + [self.r_des, self.quat_des], res
            )
        else:
            return moto.constr.create(
                "ee_constr_ineq",
                self.pos_args + [self.r_des, self.quat_des],
                cs.vcat([res, -res]),
            ).cast_ineq()

    def make_joint_limit_constr(self):
        q_min = self.fmodel.lowerPositionLimit[-self.nj :]
        q_max = self.fmodel.upperPositionLimit[-self.nj :]
        v_lim = self.fmodel.velocityLimit[-self.nj :]
        qj = self.q[-self.nj :]
        vj = self.v[-self.nj :]
        self.q_min = q_min
        self.q_max = q_max
        self.v_lim = v_lim
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

    def get_state_cost(self, terminal: bool = False):
        q_nom_res = self.q_stack - self.q_nom
        if self.is_floating_based:
            state_cost = (
                100.0 * cs.sumsqr(q_nom_res[: self.nqb])
                + 1 * cs.sumsqr(q_nom_res[self.nqb :])
                + 1.0 * cs.sumsqr(self.v_stack[:6])
                + 0.01 * cs.sumsqr(self.v_stack[6:])
            )
        else:
            state_cost = 0.1 * cs.sumsqr(q_nom_res) + 0.1 * cs.sumsqr(self.v_stack)
        state_args = self.pos_args + self.vel_args
        cost = moto.cost.create(
            "c", state_args + [self.q_nom], state_cost
        ).set_diag_hess()
        if terminal:
            return cost
        return cost

    def get_input_cost(self):
        input_args = self.acc_args
        input_cost = 1e-4 * cs.sumsqr(self.tq)
        return moto.cost.create("c_u", input_args, input_cost).set_diag_hess()


# dt = moto.sym.inputs("dt", 1, default_val=0.02)
dt_nom = moto.sym.params("dt_nom", 1, default_val=0.02)
dt = 0.02
display = False
ur5 = load("ur5_limited", display=display, verbose=True)
q_d = np.copy(ur5.q0)
model = pin.buildModelFromUrdf(ur5.urdf)
np.set_printoptions(precision=3, suppress=True, linewidth=200)
model = pinCasadiModel(model, dt=dt, q_nom=q_d, dense=True, use_fwd_dyn=True)

prob = moto.ocp.create()
prob.add(model.dyn)
prob.add(model.make_tq_limit_constr())
prob.add(model.make_joint_limit_constr())
prob.add(model.get_state_cost())
prob.add(model.get_input_cost())

prob_term = prob.clone()
prob_term.add_terminal(model.make_ee_pos_constr(soft=args.soft, cost=args.cost))
prob_term.add_terminal(model.get_state_cost(terminal=True))

prob.print_summary()
print("--" * 15)
# # moto.print_problem(prob_term)

N_horizon = 50

stats = []

output_file = args.output_file
with open(args.config, "r") as f:
    config = json.load(f)
from tqdm import tqdm
import time

for idx_cfg, cfg in tqdm(enumerate(config), total=len(config)):
    sqp = moto.sqp(n_job=4)
    g = sqp.graph
    n0 = g.set_head(g.add(sqp.create_node(prob)))
    n1 = g.set_tail(g.add(sqp.create_node(prob_term)))
    g.add_edge(n0, n1, N_horizon)

    n1.data.value[model.r_des] = np.array(cfg[0][:3])
    n1.data.value[model.quat_des] = np.array(cfg[0][3:7])
    if args.cost:
        n1.data.value[model.W_ee_cost] = np.ones(6) * 1e3

    def set_initial_state(data: moto.sqp.data_type):
        data.value[model.q] = np.array(cfg[1])
        data.value[model.qn] = np.array(cfg[1])

    sqp.apply_forward(set_initial_state)

    sqp.settings.ipm.mu0 = 1
    # sqp.settings.mu_method = moto.sqp.adaptive_mu_t.mehrotra_probing
    sqp.settings.ipm.mu_method = moto.sqp.adaptive_mu_t.mehrotra_predictor_corrector
    sqp.settings.ipm_conditional_corrector = True
    sqp.settings.prim_tol = 1e-3
    sqp.settings.dual_tol = 1e-3
    sqp.settings.comp_tol = 1e-3
    sqp.settings.rf.max_iters = 2
    sqp.settings.no_except = True

    benchmark.run(sqp, "", idx_cfg, cfg)

    del sqp

benchmark.dump()
