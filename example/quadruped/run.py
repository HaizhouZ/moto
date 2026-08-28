import argparse
import os
import time
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import moto
import numpy as np

from example.helpers import (
    ViserRobot,
    animate_trajectory,
    build_floating_base_model,
)
from example.quadruped.lifted_contact_elimination import (
    configure_lifted_contact_elimination,
)
from example.quadruped.model import QuadrupedModel, add_gait_phases, swing_feet
from example_robot_data import load


def main(*, lifted_contact_default=False):
    parser = argparse.ArgumentParser(
        description="Run the quadruped trajectory optimization example"
    )
    parser.add_argument(
        "--display", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--n-job", type=int, default=6)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--nodes-per-step", type=int, default=20)
    parser.add_argument(
        "--configuration-velocity",
        choices=("predicted", "next"),
        default="predicted",
        help="velocity used to integrate configuration; 'next' selects semi-implicit Euler",
    )
    parser.add_argument(
        "--acceleration-control",
        action="store_true",
        help=(
            "kinematic gait with acceleration input and semi-implicit Euler; "
            "no force, torque, gravity, ABA, or RNEA"
        ),
    )
    parser.add_argument(
        "--lifted-contact",
        action=argparse.BooleanOptionalAction,
        default=lifted_contact_default,
        help="lift contact force and eliminate it through the supplied graph",
    )
    parser.add_argument(
        "--explicit-rnea-contact",
        action="store_true",
        help="use explicit contact-force input with RNEA and contact constraints",
    )
    parser.add_argument(
        "--lifted-acceleration",
        action="store_true",
        help="also lift acceleration in the lifted-contact formulation",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=(
            int(os.environ["MOTO_SQP_MAX_ITER"])
            if "MOTO_SQP_MAX_ITER" in os.environ
            else None
        ),
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
    if args.acceleration_control and (
        args.explicit_rnea_contact or args.lifted_acceleration
    ):
        parser.error(
            "--acceleration-control is exclusive with contact formulation options"
        )
    if args.explicit_rnea_contact and args.lifted_acceleration:
        parser.error("--explicit-rnea-contact and --lifted-acceleration are exclusive")
    lifted_contact = (
        args.lifted_contact
        and not args.acceleration_control
        and not args.explicit_rnea_contact
    )
    if args.lifted_acceleration and not lifted_contact:
        parser.error("--lifted-acceleration requires --lifted-contact")

    dt_nom = moto.sym.params("dt_nom", 1, default_val=0.02)
    dt = 0.02
    display_env = os.getenv("MOTO_DISPLAY")
    display = args.display
    if display is None:
        display = display_env != "0" if display_env is not None else False
    profile_sqp = args.profile
    go2 = load("go2", display=False, verbose=True)
    q_d = np.copy(go2.q0)
    model = build_floating_base_model(go2.urdf)
    np.set_printoptions(precision=3, suppress=True, linewidth=200)
    model = QuadrupedModel(
        model,
        dt=dt,
        q_nom=q_d,
        use_forward_dynamics=(
            not args.acceleration_control
            and not lifted_contact
            and not args.explicit_rnea_contact
        ),
        configuration_velocity=(
            "next"
            if args.acceleration_control or lifted_contact or args.explicit_rnea_contact
            else args.configuration_velocity
        ),
        acceleration_control=args.acceleration_control,
        lifted_contact=lifted_contact,
        lifted_acceleration=args.lifted_acceleration,
    )
    if model.lifted_contact:
        configure_lifted_contact_elimination(model)
    model.prepare_problem_terms()
    print(
        "dynamics mode: "
        + (
            "acceleration-only semi-implicit Euler"
            if args.acceleration_control
            else (
                "lifted acceleration/RNEA/contact + semi-implicit Euler"
                if args.lifted_acceleration
                else (
                    "lifted RNEA/contact + semi-implicit Euler"
                    if model.lifted_contact
                    else (
                        "explicit RNEA/contact + semi-implicit Euler"
                        if args.explicit_rnea_contact
                        else f"contact dynamics ({args.configuration_velocity} configuration velocity)"
                    )
                )
            )
        )
    )

    stage_proto = model.create_stage(dt_nom)

    N_horizon = args.horizon

    # setup gait
    steps = args.steps
    nodes_per_step = args.nodes_per_step
    sqp = moto.sqp(n_job=args.n_job)
    try:
        head_stance, tail_stance = add_gait_phases(
            sqp,
            stage_proto,
            model,
            horizon=N_horizon,
            steps=steps,
            nodes_per_step=nodes_per_step,
        )
    except ValueError as error:
        parser.error(str(error))
    print(
        f"stance_length: {head_stance}+{tail_stance}, nodes_per_step: {nodes_per_step}"
    )

    model.add_terminal_terms(sqp.ed)
    nodes = sqp.nodes

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
        for index, node in enumerate(nodes):
            prob = node.prob
            print(
                f"  node[{index}] "
                f"x={prob.dim(moto.field.field___x)} "
                f"u={prob.dim(moto.field.field___u)} "
                f"y={prob.dim(moto.field.field___y)}"
            )
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

    max_update_iter = (
        args.max_iter
        if args.max_iter is not None
        else (50 if args.acceleration_control else 2)
    )
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
        if ref_node_idx >= head_stance:
            ref_step = min(steps, 1 + (ref_node_idx - head_stance) // nodes_per_step)
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
        if model.lifted_contact:
            active = np.ones(len(model.contacts.impulses))
            if ref_step > 0:
                active[list(swing_feet(ref_step))] = 0.0
            for value, parameter in zip(active, model.contact_activation):
                data.value[parameter] = value

    for index, node in enumerate(nodes):
        gait_setup(node, index)

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
    print(f"result: {res.result}, solved={res.solved}, iterations={res.num_iter}")
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

    q_res = [np.array(node.value[model.q], copy=True) for node in nodes]
    q_res.append(np.array(nodes[-1].value[model.qn], copy=True))
    dt_res = [float(dt)] * len(nodes)
    if not display:
        return

    viewer = ViserRobot(go2.urdf, floating_base=True)
    colors = [(0, 255, 0), (255, 0, 0)]
    for index, target in enumerate(cfg):
        viewer.add_target(str(index), np.r_[target[:2], 0.3], color=colors[index])
    animate_trajectory(viewer, q_res, dt_res)


if __name__ == "__main__":
    main()
