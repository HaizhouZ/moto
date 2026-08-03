import argparse
import os
import time
import sys
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
    add_stage_segments,
    add_terms,
    add_time_step_regularization,
    build_floating_base_model,
    collect_state_trajectory,
    print_graph_layout,
    solver_nodes,
    visit_nodes,
)
from example_robot_data import load


class QuadrupedModel(ContactRobotModel):
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
            configuration_velocity="predicted",
        )

    def get_state_cost(self):
        q_stack = self.q_stack
        v_stack = self.v_stack
        q_nom_res = self.q.symbolic_difference(self.q.sx, self.q_nom.sx)
        residual = cs.vcat([q_nom_res, v_stack])
        weight = np.r_[
            np.full(6, 200.0),
            np.full(q_nom_res.numel() - 6, 2.0),
            np.full(6, 2.0),
            np.full(v_stack.numel() - 6, 0.02),
        ]
        cost = moto.cost.from_vector("c", residual, weight=weight)
        return cost


def main():
    parser = argparse.ArgumentParser(
        description="Run the quadruped trajectory optimization example"
    )
    parser.add_argument(
        "--display", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--n-job", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--nodes-per-step", type=int, default=20)
    parser.add_argument(
        "--max-iter",
        type=int,
        default=int(os.getenv("MOTO_SQP_MAX_ITER", "2")),
    )
    parser.add_argument(
        "--bench-runs",
        type=int,
        default=int(os.getenv("MOTO_SQP_BENCH_RUNS", "1")),
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        default=os.getenv("MOTO_PROFILE_SQP") is not None,
    )
    args = parser.parse_args()

    dt_nom = moto.sym.params("dt_nom", 1, default_val=0.02)
    dt = 0.02
    display_env = os.getenv("MOTO_DISPLAY")
    display = args.display
    if display is None:
        display = display_env != "0" if display_env is not None else False
    profile_sqp = args.profile
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
    model = build_floating_base_model(go2.urdf)
    np.set_printoptions(precision=3, suppress=True, linewidth=200)
    model = QuadrupedModel(model, dt=dt, q_nom=q_d, use_fwd_dyn=True)
    model.joint_limit_constr = model.joint_limit_constraint(model.q, model.v)
    model.torque_limit_constr = model.torque_limit_constraint(model.tq)
    model.state_cost = model.get_state_cost()

    def build_stage_prob(robot: QuadrupedModel):
        stage_prob = moto.stage()
        add_terms(
            stage_prob,
            robot.dyn,
            robot.torque_limit_constr,
            robot.input_cost(),
        )
        robot.contacts.add_to_stage(stage_prob)
        add_time_step_regularization(stage_prob, robot.dt, dt_nom)
        add_terms(stage_prob.st, robot.joint_limit_constr, robot.state_cost)
        return stage_prob

    def add_end_node_terms(node, robot: QuadrupedModel):
        robot.contacts.add_to_endpoint(node)
        add_terms(node, robot.joint_limit_constr, robot.state_cost)

    stage_proto = build_stage_prob(model)

    N_horizon = args.horizon

    # setup gait
    steps = args.steps
    nodes_per_step = args.nodes_per_step
    total_gait_steps = steps * nodes_per_step
    if total_gait_steps > N_horizon:
        parser.error("horizon must be at least steps * nodes-per-step")
    stance_length = int((N_horizon - total_gait_steps) / 2)
    print(f"stance_length: {stance_length}, nodes_per_step: {nodes_per_step}")

    gait_setting = [1, 1, 0, 0]
    sqp = moto.sqp(n_job=args.n_job)

    def create_phase_config(step):
        constr_to_disable = []
        for idx, f in enumerate([0, 3, 1, 2]):
            if step % 2 == 0:
                if not gait_setting[idx]:
                    constr_to_disable.append(model.contacts.impulses[f])
            else:
                if gait_setting[idx]:
                    constr_to_disable.append(model.contacts.impulses[f])
        return moto.active_status_config(deactivate_list=constr_to_disable)

    segment_lengths = [stance_length]
    segment_lengths.extend([nodes_per_step] * steps)
    segment_lengths.append(stance_length)

    segment_start_nodes = [stage_proto]
    segment_start_nodes.extend(
        stage_proto.with_status(create_phase_config(step)) for step in range(1, steps + 1)
    )
    segment_start_nodes.append(stage_proto.copy())
    graph_stages = add_stage_segments(sqp, segment_start_nodes, segment_lengths)

    add_end_node_terms(graph_stages[-1].ed, model)
    nodes = solver_nodes(sqp)

    if os.getenv("MOTO_DEBUG_SOLVER_PROBS"):
        print("--" * 15)
        print("Stage prototype:")
        stage_proto.print_summary()
        print("Head solver node problem:")
        nodes[0].prob.print_summary()
        print("Tail solver node problem:")
        nodes[-1].prob.print_summary()

    if os.getenv("MOTO_DEBUG_GRAPH_LAYOUT"):
        print("Flattened solver graph layout:")
        print_graph_layout(nodes)
    sqp.settings.ipm.mu0 = 1.0
    sqp.settings.ipm.mu_method = moto.sqp.adaptive_mu_t.monotonic_decrease
    sqp.settings.ipm_conditional_corrector = True
    sqp.settings.prim_tol = 1e-3
    sqp.settings.dual_tol = 1e-3
    sqp.settings.comp_tol = 1e-3
    sqp.settings.rf.max_iters = 2
    sqp.settings.ls.update_alpha_dual = False
    sqp.settings.restoration.enabled = True
    sqp.settings.restoration.max_iter = 10
    sqp.settings.restoration.rho_eq = 1e-6
    sqp.settings.ls.primal_gamma = 1e-4
    sqp.settings.ls.method = moto.sqp.search_method_filter

    max_update_iter = args.max_iter
    bench_runs = args.bench_runs
    bench_show_last = os.getenv("MOTO_SQP_BENCH_SHOW_LAST", "1") != "0"
    print(f"SQP update iters: {max_update_iter}")
    print(f"SQP benchmark runs: {bench_runs}")

    cfg = [
        [-0.8181818181818181, 0.3737373737373739],
        [0.4545454545454546, -0.21212121212121204],
    ]
    print("")
    sys.stdout.flush()

    def gait_setup(data: moto.sqp.data_type, node_index):
        ref_node_idx = min(node_index + 1, N_horizon)
        ref_step = 0
        if ref_node_idx >= stance_length:
            ref_step = min(steps, 1 + (ref_node_idx - stance_length) // nodes_per_step)
        node_progress = ref_node_idx / N_horizon
        if 1 <= ref_step <= 2:
            data.value[model.q_nom][0] = node_progress * cfg[0][0]
            data.value[model.q_nom][1] = node_progress * cfg[0][1]
        elif ref_step > 2:
            data.value[model.q_nom][0] = cfg[0][0] + node_progress * (
                cfg[1][0] - cfg[0][0]
            )
            data.value[model.q_nom][1] = cfg[0][1] + node_progress * (
                cfg[1][1] - cfg[0][1]
            )

    visit_nodes(nodes, gait_setup)

    cnt = 0
    iters = 0
    start = time.perf_counter()
    res = None
    for i in range(bench_runs):
        bench_verbose = os.getenv("MOTO_SQP_BENCH_VERBOSE") is not None or (
            bench_show_last and i + 1 == bench_runs
        )
        res = sqp.update(max_update_iter, verbose=bench_verbose, profile=profile_sqp)
        sqp.settings.ipm.warm_start = True
        cnt += 1
        iters += res.num_iter
    elapsed = time.perf_counter() - start

    sys.stdout.flush()
    print(f"sqp.update() took {elapsed / cnt:.3f} seconds")
    if iters > 0:
        print(f"per iteration took {elapsed / iters * 1000:.3f} ms")

    if profile_sqp:
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

    q_res, dt_res = collect_state_trajectory(nodes, model.q, model.qn, dt)
    if not display:
        return

    import meshcat.transformations as tf
    import meshcat_shapes as mcs

    viz = go2.viz

    color = [0x00FF00, 0xFF0000]
    for i in range(2):
        mcs.point(viz.viewer[f"/target{i}"], color=color[i], radius=0.04)
        pose = tf.compose_matrix(translate=cfg[i][:2] + [0.3])
        viz.viewer[f"/target{i}"].set_transform(pose)
        mcs.frame(viz.viewer[f"/frame{i}"])
        viz.viewer[f"/frame{i}"].set_transform(pose)

    while True:
        for i in range(len(q_res)):
            start = time.perf_counter()
            go2.display(q_res[i])
            if i != N_horizon:
                dt_ = dt_res[i]
                remaining = dt_ - (time.perf_counter() - start)
                if remaining > 0:
                    time.sleep(remaining)
        time.sleep(0.5)


if __name__ == "__main__":
    main()
