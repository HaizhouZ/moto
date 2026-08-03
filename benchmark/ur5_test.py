import moto
import casadi as cs
import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin

from example_robot_data import load
from benchmark_tool import benchmark_tool
from example.helpers import pinocchio_states

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

    def make_ee_pos_constr(self, soft: bool = False, cost: bool = False):
        self.r_des = moto.sym.params("r_des", 3, default_val=np.zeros(3))
        self.quat_des = moto.sym.params(
            "quat_des", 4, default_val=np.array([0.0, 0.0, 0.0, 1.0])
        )
        ee_des = cpin.XYZQUATToSE3(cs.vcat([self.r_des, self.quat_des]))
        ee_pos = self.data.oMf[self.ee_id]
        res = cpin.log6(ee_des.inverse() * ee_pos).np
        if cost:
            result = moto.cost.from_vector(
                "ee_cost", res,
                weight=np.array([40.0, 40.0, 40.0, 0.0, 0.0, 0.0]),
            )
            self.W_ee_cost = result.weight
            return result
        if not soft:
            return moto.constr.create(
                "ee_constr", res
            )
        else:
            return moto.constr.create(
                "ee_constr_ineq",
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
        return moto.ineq.create(
            "q_limit",
            cs.vcat([qj, vj]),
            np.concatenate([q_min, -v_lim]),
            np.concatenate([q_max, v_lim]),
        )

    def make_tq_limit_constr(self):
        tq_limit = model.effortLimit[-self.nj :]
        return moto.ineq.bounds("tq_limit", self.tq, -tq_limit, tq_limit)

    def get_state_cost(self):
        q_nom_res = self.q.symbolic_difference(self.q.sx, self.q_nom.sx)
        if self.is_floating_based:
            weight = np.r_[
                np.full(self.nqb, 200.0),
                np.full(q_nom_res.numel() - self.nqb, 2.0),
                np.full(6, 2.0),
                np.full(self.v_stack.numel() - 6, 0.02),
            ]
        else:
            weight = np.full(q_nom_res.numel() + self.v_stack.numel(), 0.2)
        cost = moto.cost.from_vector(
            "c", cs.vcat([q_nom_res, self.v_stack]),
            weight=weight,
        )
        return cost

    def get_input_cost(self):
        return moto.cost.from_vector("c_u", self.tq, weight=2e-4)


# dt = moto.sym.inputs("dt", 1, default_val=0.02)
dt_nom = moto.sym.params("dt_nom", 1, default_val=0.02)
dt = 0.02
display = False
ur5 = load("ur5_limited", display=display, verbose=True)
q_d = np.copy(ur5.q0)
model = pin.buildModelFromUrdf(ur5.urdf)
np.set_printoptions(precision=3, suppress=True, linewidth=200)
model = pinCasadiModel(model, dt=dt, q_nom=q_d, dense=True, use_fwd_dyn=True)
joint_limit_constr = model.make_joint_limit_constr()
state_cost = model.get_state_cost()

prob = moto.stage_ocp.create()
prob.add(model.dyn)
prob.add(model.make_tq_limit_constr())
prob.st.add(joint_limit_constr)
prob.st.add(state_cost)
prob.add(model.get_input_cost())

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
    stages = sqp.add_stage(prob, N_horizon)
    stages[-1].ed.add(joint_limit_constr)
    stages[-1].ed.add(model.make_ee_pos_constr(soft=args.soft, cost=args.cost))
    stages[-1].ed.add(state_cost)
    nodes = sqp.nodes

    nodes[-1].value[model.r_des] = np.array(cfg[0][:3])
    nodes[-1].value[model.quat_des] = np.array(cfg[0][3:7])
    if args.cost:
        nodes[-1].value[model.W_ee_cost] = np.ones(6) * 1e3

    def set_initial_state(data: moto.sqp.data_type):
        data.value[model.q] = np.array(cfg[1])
        data.value[model.qn] = np.array(cfg[1])

    for node in nodes:
        set_initial_state(node)

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
