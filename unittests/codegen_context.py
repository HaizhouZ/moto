"""Native codegen isolation, including asynchronous and lazy compilation."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

import casadi as cs
import moto
import numpy as np


REPO = Path(__file__).resolve().parents[1]


def build(context, gain=1.0, *, optimized=False):
    x, y = moto.sym.states("directory_x", 1)
    u = moto.sym.inputs("directory_u", 1)
    dynamics = moto.dense_dynamics.create("directory_dynamics", y - x - gain * u)
    stage = moto.stage(codegen=context)
    stage.add([dynamics, moto.cost.from_scalar("directory_control", u)])
    solver = moto.sqp(n_job=1, codegen=context)
    solver.stages.extend([stage.copy() for _ in range(2)])
    solver.ed.add(moto.constr.create("directory_target", x - 1.0))
    if optimized:
        solver.settings.initial_state = moto.sqp.initial_state_mode.optimized
    return solver, stage, x, y, u


def solve(model):
    solver, _, x, y, u = model
    nodes = solver.nodes
    result = solver.update(20, verbose=False)
    assert result.solved, result.result
    np.testing.assert_allclose(nodes[-1].value[y], [1.0], atol=1e-6)
    return np.array([node.value[u][0] for node in nodes])


def binaries(directory):
    return {p.relative_to(directory): p.stat().st_mtime_ns
            for p in Path(directory).rglob("*.so")}


class CodegenContextTests(unittest.TestCase):
    def setUp(self):
        self.original_cwd = Path.cwd()
        self.temp = tempfile.TemporaryDirectory(prefix="moto_codegen_")
        self.root = Path(self.temp.name)
        os.chdir(self.root)

    def tearDown(self):
        os.chdir(self.original_cwd)
        self.temp.cleanup()

    def test_async_instances_and_live_caches_are_isolated(self):
        contexts = [moto.codegen_context(name) for name in ("a space", "b", "c")]
        # Queue matching function names in several roots before waiting for any.
        models = [build(ctx, gain) for ctx, gain in zip(contexts, (1., 1., 2.))]
        for model in models:
            model[0].nodes
        with ThreadPoolExecutor(max_workers=3) as pool:
            controls = list(pool.map(solve, models))
        for values, expected in zip(controls, (0.5, 0.5, 0.25)):
            np.testing.assert_allclose(values, expected, atol=1e-6)
        for ctx in contexts:
            self.assertTrue(list(Path(ctx.output_dir).glob(".moto_artifacts/**/*.so")))
            self.assertTrue(list(Path(ctx.linear_dir).glob("*.so")))
        self.assertFalse((self.root / "gen").exists())
        before = binaries(contexts[0].output_dir)
        repeat = build(moto.codegen_context(self.root / "a space" / "."))
        np.testing.assert_allclose(solve(repeat), 0.5, atol=1e-6)
        self.assertEqual(before, binaries(contexts[0].output_dir))

    def test_context_is_absolute_immutable_and_survives_cwd_change(self):
        ctx = moto.codegen_context(Path("relative folder"))
        self.assertEqual(Path(ctx.output_dir), self.root / "relative folder")
        with self.assertRaises(AttributeError):
            ctx.output_dir = "elsewhere"
        model = build(ctx)
        (self.root / "elsewhere").mkdir()
        os.chdir(self.root / "elsewhere")
        solve(model)
        self.assertFalse(Path("gen").exists())
        self.assertTrue(binaries(ctx.output_dir))
        with self.assertRaises(ValueError):
            moto.codegen_context("")
        Path("file").write_text("not a directory")
        with self.assertRaises((RuntimeError, OSError)):
            moto.codegen_context(Path("file") / "child")

    def test_default_and_explicit_finalization(self):
        stage = moto.stage()
        solver = moto.sqp(1)
        self.assertEqual(stage.codegen.output_dir, str(self.root / "gen"))
        self.assertEqual(solver.codegen.output_dir, stage.codegen.output_dir)
        context = moto.codegen_context("explicit")
        x, _ = moto.sym.states("explicit_x", 1)
        term = moto.cost.from_scalar("explicit_cost", x)
        term.finalize(codegen=context)
        self.assertEqual(x.codegen.output_dir, context.output_dir)
        self.assertTrue(binaries(context.output_dir))

    def test_cross_directory_dependencies_and_stages_are_rejected(self):
        a, b = moto.codegen_context("a"), moto.codegen_context("b")
        x, _ = moto.sym.states("bound_x", 1)
        x.finalize(codegen=a)
        term = moto.cost.from_scalar("must_not_compile", x)
        with self.assertRaisesRegex(ValueError, "fresh model"):
            moto.stage(codegen=b).add(term)
        self.assertIsNone(term.codegen)
        self.assertFalse(binaries(b.output_dir))
        stage = moto.stage(codegen=a)
        stage.add(term)
        stage.wait_until_ready()
        y, _ = moto.sym.states("bound_y", 1)
        y.finalize(codegen=b)
        with self.assertRaisesRegex(ValueError, "fresh model"):
            term.reuse_remap([(x, y)])
        solver = moto.sqp(codegen=b)
        solver.stages.append(stage.copy())
        with self.assertRaisesRegex(ValueError, "fresh model"):
            solver.nodes
        self.assertFalse((self.root / "gen").exists())

    def test_canonical_precompute_and_remaps_stay_in_context(self):
        for name in ("canonical_a", "canonical_b"):
            context = moto.codegen_context(name)
            calls = []
            x, _ = moto.sym.states("canonical_x", 1)
            z, _ = moto.sym.states("canonical_z", 1)

            def outputs():
                calls.append("precompute")
                return [x * x]

            values = moto.precompute.canonical("directory_pre", 1, [x], outputs,
                                                codegen=context)
            remapped = moto.precompute.canonical("directory_pre", 1, [z], outputs,
                                                  codegen=context)

            def factory(function_name):
                calls.append("function")
                return moto.constr.create(function_name, values[0])

            source = moto.func.canonical("directory_fun", 1, values, factory,
                                         codegen=context)
            other = moto.func.canonical("directory_fun", 1, remapped, factory,
                                        codegen=context)
            self.assertEqual(calls, ["precompute", "function"])
            self.assertEqual(source.codegen.output_dir, context.output_dir)
            self.assertEqual(other.codegen.output_dir, context.output_dir)
            stage = moto.stage(codegen=context)
            stage.ed.add(other)
            stage.wait_until_ready()
            self.assertTrue(binaries(context.output_dir))
        self.assertFalse((self.root / "gen").exists())

    def test_rejected_remap_does_not_bind_source_or_fresh_targets(self):
        a, b = moto.codegen_context(), moto.codegen_context("other")
        for bind_source in (False, True):
            for reverse in (False, True):
                with self.subTest(bind_source=bind_source, reverse=reverse):
                    u = moto.sym.inputs("remap_u", 1)
                    v = moto.sym.inputs("remap_v", 1)
                    fresh = moto.sym.inputs("remap_fresh", 1)
                    foreign = moto.sym.inputs("remap_foreign", 1)
                    foreign.finalize(codegen=b)
                    term = moto.constr.create("atomic_remap", u + v)
                    if bind_source:
                        term._bind_codegen(a)
                    targets = (foreign, fresh) if reverse else (fresh, foreign)
                    with self.assertRaisesRegex(ValueError, "fresh model"):
                        term.reuse_remap(list(zip((u, v), targets)))
                    self.assertIsNone(fresh.codegen)
                    self.assertEqual(foreign.codegen.output_dir, b.output_dir)
                    for expression in (term, u, v):
                        if bind_source:
                            self.assertEqual(expression.codegen.output_dir, a.output_dir)
                        else:
                            self.assertIsNone(expression.codegen)
                    # A failed attempt leaves the fresh target usable elsewhere.
                    fresh.finalize(codegen=b)
        self.assertFalse(binaries(a.output_dir))
        self.assertFalse(binaries(b.output_dir))

    def test_manifold_helpers_and_structured_euler_use_context(self):
        context = moto.codegen_context("manifold")
        q = cs.SX.sym("q", 2)
        dq = cs.SX.sym("dq", 1)
        other = cs.SX.sym("other", 2)
        angle = cs.atan2(q[1], q[0])
        integrated = cs.vertcat(cs.cos(angle + dq), cs.sin(angle + dq))
        difference = cs.atan2(q[0] * other[1] - q[1] * other[0], cs.dot(q, other))
        x, y = moto.casadi_manifold.create("directory_circle", q, dq, integrated,
                                          other, difference, np.array([1., 0.]))
        u = moto.sym.inputs("circle_u", 1)
        dynamics = moto.semi_implicit_euler.create(
            "directory_euler", x.symbolic_difference(x.symbolic_integrate(x, u), y),
            moto.semi_implicit_euler.state.pos)
        stage = moto.stage(codegen=context)
        stage.add([dynamics, moto.cost.from_scalar("circle_control", u)])
        solver = moto.sqp(1, codegen=context)
        solver.stages.append(stage.copy())
        result = solver.update(1, verbose=False)
        self.assertTrue(result.solved)
        for suffix in ("integrate", "difference"):
            self.assertTrue(list(Path(context.output_dir).glob(
                f".moto_artifacts/directory_circle_{suffix}/**/*.so")))
        self.assertTrue(list(Path(context.linear_dir).glob("*.so")))
        self.assertFalse((self.root / "gen").exists())

    def test_optimized_initial_state_and_derived_problems(self):
        context = moto.codegen_context("optimized")
        model = build(context, optimized=True)
        np.testing.assert_allclose(solve(model), 0.0, atol=1e-6)
        np.testing.assert_allclose(model[0].nodes[0].value[model[2]], [1.], atol=1e-6)
        for example in ("restoration", "lifted_sparse_elimination"):
            spec = importlib.util.spec_from_file_location(
                f"context_{example}", REPO / "example" / "toy" / f"{example}.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            context = moto.codegen_context(example)
            solver = module.main(codegen=context)
            if solver is not None:
                solver.update(1, verbose=False)
            self.assertTrue(binaries(context.output_dir))
            self.assertTrue(list(Path(context.linear_dir).glob("*.so")))
        self.assertFalse((self.root / "gen").exists())


if __name__ == "__main__":
    unittest.main()
