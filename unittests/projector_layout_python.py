import importlib.util
import site
import sys
import sysconfig
from pathlib import Path


def load_local_moto():
    for key in ("purelib", "platlib"):
        path = sysconfig.get_paths().get(key)
        if path:
            site.addsitedir(path)
    root = Path(__file__).resolve().parents[1]
    source_bindings = root / "bindings"
    build_bindings = root / "build" / "bindings"
    spec = importlib.util.spec_from_file_location(
        "moto",
        source_bindings / "__init__.py",
        submodule_search_locations=[str(source_bindings), str(build_bindings)],
    )
    module = importlib.util.module_from_spec(spec)
    module.__path__ = [str(source_bindings), str(build_bindings)]
    sys.modules["moto"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


moto = load_local_moto()


def names(prob, field):
    return [expr.name for expr in prob.exprs(field)]


def main():
    order_second = moto.approx_order.approx_order_second
    field = moto.field

    x, xn = moto.sym.states("x_py_projector", 1)
    u0 = moto.sym.inputs("u0_py_projector", 1)
    u1 = moto.sym.inputs("u1_py_projector", 1)

    dyn = moto.dense_dynamics.create(
        "dyn_py_projector",
        [x, xn, u0, u1],
        xn.sx - x.sx - u0.sx - 2.0 * u1.sx,
        order_second,
    )

    node_prob = moto.node_ocp.create()
    eq_state = moto.constr.create(
        "eq_state_py_projector",
        [x],
        x.sx,
        order_second,
        field.field___eq_x,
    )
    node_prob.add(eq_state)
    state_group = node_prob.projector().group()
    state_group.require_constraint([eq_state])

    edge_prob = moto.edge_ocp.create()
    eq_u0 = moto.constr.create(
        "eq_u0_py_projector",
        [u0],
        u0.sx,
        order_second,
        field.field___eq_xu,
    )
    eq_u1 = moto.constr.create(
        "eq_u1_py_projector",
        [u1],
        u1.sx,
        order_second,
        field.field___eq_xu,
    )
    edge_prob.add(dyn)
    edge_prob.add(eq_u0)
    edge_prob.add(eq_u1)

    proj = edge_prob.projector()
    drive_group = proj.group()
    drive_group.require_primal([u1])
    drive_group.require_constraint([eq_u1])
    balance_group = proj.group()
    balance_group.require_primal([u0])
    balance_group.require_constraint([eq_u0])
    drive_group.require_before(state_group)
    state_group.require_before(balance_group)

    modeled = moto.graph_model()
    n0 = modeled.create_node(node_prob)
    n1 = modeled.create_node()
    e01 = modeled.connect(n0, n1, edge_prob)

    composed = modeled.compose(e01)

    assert names(composed, field.field___u) == [
        "u1_py_projector",
        "u0_py_projector",
    ]
    assert names(composed, field.field___eq_xu) == [
        "eq_u1_py_projector",
        "eq_u0_py_projector",
    ]

    blocks = composed.compiled_hard_constraint_blocks
    assert len(blocks) == 3
    assert [block.group_id for block in blocks] == [
        drive_group.id,
        state_group.id,
        balance_group.id,
    ]
    assert [block.field for block in blocks] == [
        field.field___eq_xu,
        field.field___eq_x,
        field.field___eq_xu,
    ]


if __name__ == "__main__":
    main()
