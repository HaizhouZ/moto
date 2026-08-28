import argparse
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

from example.helpers import (
    PinocchioCasadiModel,
    ViserRobot,
    animate_trajectory,
    frame_placement,
)
from example_robot_data import load


class ArmModel(PinocchioCasadiModel):
    def __init__(
        self,
        model: pin.Model,
        name: str = "",
        dt: cs.SX | float = 0.01,
        q_nom: np.ndarray | None = None,
        use_fwd_dyn: bool = False,
    ):
        super().__init__(model, name)
        self.use_fwd_dyn = use_fwd_dyn
        self.ee_frame = "ee_link"
        self.ee_id = model.getFrameId(self.ee_frame)
        self.create_trajectory_variables(dt, q_nom)

        def implicit_euler():
            q_next = self.q.symbolic_integrate(self.q.sx, self.vn * dt)
            return cs.vcat([self.q.symbolic_difference(self.qn.sx, q_next)])

        self.joint_euler = implicit_euler()
        self.update_kinematics(self.q_stack, self.v_stack)

        if self.use_fwd_dyn:
            self.aba = self.forward_dynamics_step(
                self.q_stack, self.v_stack, self.tq, dt
            )
        else:
            self.rnea = self.inverse_dynamics_impulse(
                self.q_stack, self.v_stack, self.a_stack, dt
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
                "arm_" + self.name + "_fd", cs.vcat(out + [self.vn - v_next])
            )
        else:
            tau = self.generalized_torque(self.tq)
            return moto.dense_dynamics.create(
                "arm_" + self.name + "_id",
                cs.vcat(out + [self.rnea - tau * self.dt]),
            )

    def make_ee_pos_constr(self):
        if not hasattr(self, "r_des"):
            self.r_des = moto.sym.params("r_des", 3, default_val=np.zeros(3))
            self.quat_des = moto.sym.params(
                "quat_des", 4, default_val=np.array([0.0, 0.0, 0.0, 1.0])
            )
        ee_des = cpin.XYZQUATToSE3(cs.vcat([self.r_des, self.quat_des]))
        ee_pos = self.data.oMf[self.ee_id]
        c: moto.pmm_constr = moto.constr.create(
            "arm_ee_constr",
            cpin.log6(ee_pos.inverse() * ee_des).np,
        )
        return c

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
            "arm_state_cost",
            cs.vcat([q_nom_res, self.v_stack]), weight=weight
        )
        return cost

    def get_input_cost(self):
        return moto.cost.from_vector("arm_input_cost", self.tq, weight=2e-4)

    def get_dt_reg(self, dt_nom):
        if not isinstance(self.dt, cs.SX):
            raise ValueError("dt is not a symbolic variable")
        return [
            moto.cost.from_scalar(
                "arm_dt_reg", self.dt - dt_nom, weight=2e3
            ),
            moto.ineq.bounds("arm_dt_bound", self.dt, 1e-2, 0.1),
        ]


# dt = moto.sym.inputs("dt", 1, default_val=0.02)
dt_nom = moto.sym.params("dt_nom", 1, default_val=0.02)
dt = 0.02


def build_sqp(n_job: int = 4):
    ur5 = load("ur5_limited", display=False, verbose=True)
    q_d = np.copy(ur5.q0)
    model = pin.buildModelFromUrdf(ur5.urdf)
    np.set_printoptions(precision=3, suppress=True, linewidth=200)
    model = ArmModel(model, dt=dt, q_nom=q_d, use_fwd_dyn=False)
    joint_limit_constr = model.joint_limit_constraint(
        model.q, model.v, name="arm_q_limit"
    )
    torque_limit_constr = model.torque_limit_constraint(model.tq, name="arm_tq_limit")
    state_cost = model.get_state_cost()

    stage_prob = moto.stage()
    stage_prob.add([model.dyn, torque_limit_constr, model.get_input_cost()])
    stage_prob.st.add([joint_limit_constr, state_cost])

    stage_prob.print_summary()
    print("--" * 15)

    horizon = 50
    sqp = moto.sqp(n_job=n_job)
    sqp.stages.extend([stage_prob.copy() for _ in range(horizon)])
    sqp.ed.add(
        [joint_limit_constr, state_cost, model.make_ee_pos_constr()]
    )

    cfg = [
        [
            -0.20386130721377144,
            -0.2780080314109077,
            0.05367065178434891,
            0.6564066614217499,
            -0.7069685526302121,
            0.262510077975368,
            -0.020352380560065237,
        ],
        [
            0.39077402479947176,
            0.2908006855904619,
            -0.5419511019201957,
            0.22346096937115068,
            -0.20115318796212112,
            0.3804248223817164,
        ],
    ]

    def set_initial_state(data: moto.sqp.data_type, _):
        data.value[model.q] = np.array(cfg[1])
        data.value[model.qn] = np.array(cfg[1])
        if data.prob.dim(moto.field.field___eq_x) > 0:
            data.value[model.r_des] = np.array(cfg[0][:3])
            data.value[model.quat_des] = np.array(cfg[0][3:7])
            if hasattr(model, "W_ee_cost"):
                data.value[model.W_ee_cost] = np.ones(6) * 1e8

    nodes = sqp.nodes
    for index, node in enumerate(nodes):
        set_initial_state(node, index)

    sqp.settings.ipm.mu0 = 0.1
    # sqp.settings.ipm.mu_method = moto.sqp.adaptive_mu_t.mehrotra_predictor_corrector
    sqp.settings.ipm.mu_method = moto.sqp.monotonic_decrease
    sqp.settings.prim_tol = 1e-3
    sqp.settings.dual_tol = 1e-3
    sqp.settings.comp_tol = 1e-3
    sqp.settings.restoration.rho_eq = 1e3
    sqp.settings.restoration.rho_ineq = 1e3
    sqp.settings.restoration.rho_u = 0.0001
    sqp.settings.restoration.rho_y = 0.0001
    sqp.settings.ls.update_alpha_dual = False

    return sqp, model, ur5, cfg, nodes


def visualize_solution(ur5, cfg, q_res, dt_res):
    viewer = ViserRobot(ur5.urdf)
    viewer.add_target("end_effector", cfg[0][:3], xyzw=cfg[0][3:7])
    animate_trajectory(viewer, q_res, dt_res)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--display", action="store_true", help="enable Viser visualization"
    )
    parser.add_argument(
        "--n-job", type=int, default=4, help="number of solver worker jobs"
    )
    parser.add_argument(
        "--max-iter", type=int, default=100, help="maximum SQP iterations"
    )
    args = parser.parse_args()

    sqp, model, ur5, cfg, nodes = build_sqp(n_job=args.n_job)

    import time

    sys.stdout.flush()
    start = time.perf_counter()
    kkt = sqp.update(args.max_iter)
    sys.stdout.flush()
    print(f"sqp.update({args.max_iter}) took {time.perf_counter() - start:.3f} seconds")
    print(f"result       : {kkt.result}")
    print(f"num_iter     : {kkt.num_iter}")
    print(f"prim_res     : {kkt.inf_prim_res:.3e}")
    print(f"dual_res     : {kkt.inf_dual_res:.3e}")
    print(f"comp_res     : {kkt.inf_comp_res:.3e}")
    print(f"solved       : {kkt.solved}")

    q_res = [np.array(node.value[model.q], copy=True) for node in nodes]
    q_res.append(np.array(nodes[-1].value[model.qn], copy=True))
    dt_res = [float(dt)] * len(nodes)

    eef = frame_placement(model.fmodel, q_res[-1], model.ee_id)
    eef_des = pin.XYZQUATToSE3(cfg[0])
    print("final ee pos err:", pin.log6(eef_des.inverse() * eef).np)

    if args.display:
        visualize_solution(ur5, cfg, q_res, dt_res)


if __name__ == "__main__":
    main()
