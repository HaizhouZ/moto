import argparse
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import moto
import casadi as cs
import numpy as np
import pinocchio as pin

from example.helpers import (
    ContactRobotModel,
    GO2_FOOT_FRAMES,
    add_terms,
    add_time_step_regularization,
    build_floating_base_model,
    numeric_frame_linear_kinematics,
    solver_nodes,
    visit_nodes,
)
from example_robot_data import load


class QuadrupedMpcModel(ContactRobotModel):
    def __init__(
        self,
        model: pin.Model,
        name: str = "",
        dt: cs.SX | float = 0.01,
        q_nom: np.ndarray | None = None,
        foot_frames=GO2_FOOT_FRAMES,
        use_fwd_dyn: bool = False,
    ):
        super().__init__(
            model,
            name=name,
            dt=dt,
            q_nom=q_nom,
            contact_frames=foot_frames,
            use_forward_dynamics=use_fwd_dyn,
            configuration_velocity="next",
        )

    def get_state_cost(self):
        q_nom_res = self.q.symbolic_difference(self.q.sx, self.q_nom.sx)
        residual = cs.vcat([q_nom_res, self.v_stack])
        weight = np.r_[
            np.full(6, 200.0),
            np.full(q_nom_res.numel() - 6, 2.0),
            np.full(6, 2.0),
            np.full(self.v_stack.numel() - 6, 0.02),
        ]
        cost = moto.cost.from_vector("c_mpc", residual, weight=weight)
        return cost


def main():
    parser = argparse.ArgumentParser(description="Run the quadruped MuJoCo MPC demo")
    parser.add_argument("--n-job", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--warm-start-iter", type=int, default=100)
    parser.add_argument("--mpc-iter", type=int, default=5)
    parser.add_argument("--control-frequency", type=float, default=10.0)
    parser.add_argument(
        "--scene",
        type=Path,
        default=_REPO_ROOT / "example/quadruped/rsc/scene.xml",
        help="MuJoCo scene containing the Go2 model and actuators",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if not args.scene.is_file():
        parser.error(f"MuJoCo scene not found: {args.scene}")
    if args.control_frequency <= 0:
        parser.error("control-frequency must be positive")

    import mujoco
    import mujoco.viewer

    dt_nom = moto.sym.params("dt_nom", 1, default_val=0.02)
    dt = 0.02
    go2 = load("go2", display=False, verbose=True)
    q_d = np.copy(go2.q0)
    q_d[2] -= 0.02
    model = build_floating_base_model(go2.urdf)
    np.set_printoptions(precision=3, suppress=True, linewidth=200)
    model = QuadrupedMpcModel(model, dt=dt, q_nom=q_d, use_fwd_dyn=True)
    model.joint_limit_constr = model.joint_limit_constraint(model.q, model.v)
    model.torque_limit_constr = model.torque_limit_constraint(model.tq)
    model.state_cost = model.get_state_cost()

    prob = moto.stage()
    add_terms(prob, model.dyn, model.torque_limit_constr, model.input_cost())
    model.contacts.add_to_stage(prob)
    add_terms(prob.st, model.joint_limit_constr, model.state_cost)
    add_time_step_regularization(prob, model.dt, dt_nom)

    prob.print_summary()
    print("--" * 15)

    N_horizon = args.horizon

    sqp = moto.sqp(n_job=args.n_job)
    stages = sqp.add_stage(prob, N_horizon)
    model.contacts.add_to_endpoint(stages[-1].ed)
    add_terms(stages[-1].ed, model.joint_limit_constr, model.state_cost)

    sqp.settings.ipm.mu0 = 0.1
    sqp.settings.ipm.mu_method = moto.sqp.adaptive_mu_t.mehrotra_predictor_corrector
    sqp.settings.ipm_conditional_corrector = True
    sqp.settings.prim_tol = 1e-3
    sqp.settings.dual_tol = 1e-3
    sqp.settings.comp_tol = 1e-3
    sqp.settings.rf.max_iters = 2

    def stance_ref(data: moto.sqp.data_type, node_index):
        # periodic base pose reference (smooth oscillation + slow drift between cfg waypoints)
        freq = 1  # Hz
        phase = 2 * np.pi * freq * (current_time + node_index * model.dt)

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

    mj_model = mujoco.MjModel.from_xml_path(str(args.scene))
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
    current_time = 0.0
    nodes = solver_nodes(sqp)
    visit_nodes(nodes, stance_ref)
    control_freq = args.control_frequency
    update_interval = max(1, round(1 / (control_freq * model.dt)))
    if len(nodes) <= update_interval:
        parser.error("horizon is too short for the requested control-frequency")
    n0 = nodes[0]
    data = go2.model.createData()
    model.contacts.set_kinematic_gain(nodes, 0, count=10)
    sys.stdout.flush()
    sqp.update(args.warm_start_iter, verbose=True)
    sys.stdout.flush()
    start = time.perf_counter()
    sqp.settings.ipm.warm_start = True

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        while viewer.is_running():
            loop_start = time.perf_counter()
            # Update initial state for MPC
            n0.value[model.q] = mj_data.qpos.copy()
            # convert mujoco quaternion to pinocchio format
            n0.value[model.q][3:7] = np.array(
                [mj_data.qpos[4], mj_data.qpos[5], mj_data.qpos[6], mj_data.qpos[3]]
            )
            if args.verbose:
                print("Current base:", n0.value[model.q])
            n0.value[model.v] = mj_data.qvel.copy()
            if args.verbose:
                print("Current base velocity:", n0.value[model.v])

            # Update reference trajectory
            current_time = mj_data.time
            visit_nodes(nodes, stance_ref)
            if args.verbose:
                print(f"Updated reference trajectory for {len(nodes)} nodes")
            # Run MPC iteration
            mpc_st = time.perf_counter()
            res = sqp.update(args.mpc_iter, verbose=False)
            mpc_ed = time.perf_counter()
            print(
                f"MPC iteration {res.num_iter}, prim_res: {res.inf_prim_res:.3e}, dual res: {res.inf_dual_res:.3e}, timing: {(mpc_ed - mpc_st) * 1000:.3f} ms"
            )
            cnt += 1
            iters += res.num_iter

            # Extract and apply control.

            if args.verbose:
                print(
                    "n0 reaction forces:",
                    [n0.value[f] / dt for f in model.contacts.impulses],
                )
                for i, n in enumerate(nodes[0:5]):
                    print(f"n{i} base state:\t\t", n.value[model.q][:7])
                    print(f"n{i} base velocity:\t", n.value[model.v][:6])
                    heights, velocities = numeric_frame_linear_kinematics(
                        go2.model,
                        data,
                        n.value[model.q],
                        n.value[model.v],
                        model.contacts.frame_ids,
                    )
                    print("foot heights:", heights)
                    print("foot velocities:", velocities)
                    print("torques:", n.value[model.tq])
                    print(
                        "reaction forces:",
                        [n.value[f] / dt for f in model.contacts.impulses],
                    )
            step = 0
            while step < update_interval:
                current_node_idx = int(step * sim_dt // dt + 1)
                mj_data.ctrl[:] = nodes[current_node_idx].value[model.tq]
                qj_mujoco = mj_data.qpos[7:]
                qj_mpc = nodes[current_node_idx].value[model.q][7:]
                vj_mujoco = mj_data.qvel[6:]
                vj_mpc = nodes[current_node_idx].value[model.v][6:]
                mj_data.ctrl[:] += 50.0 * (qj_mpc - qj_mujoco) + 0.5 * (
                    vj_mpc - vj_mujoco
                )
                mujoco.mj_step(mj_model, mj_data)
                step += 1
            viewer.sync()

            remaining = update_interval * sim_dt - (time.perf_counter() - loop_start)
            if remaining > 0:
                time.sleep(remaining)

    print(
        f"sqp.update() took {(time.perf_counter() - start) / max(1, cnt):.3f} seconds"
    )
    print(
        f"per iteration took {(time.perf_counter() - start) / max(1, iters) * 1000:.3f} ms"
    )


if __name__ == "__main__":
    main()
