PUBLIC_BINDINGS = {
    "active_status_config",
    "approx_order",
    "constr",
    "cost",
    "dense_dynamics",
    "edge_ocp",
    "expr",
    "field",
    "func",
    "ineq",
    "node_ocp",
    "pmm_constr",
    "quaternion",
    "sym",
}


def export_public_bindings(extension, namespace):
    for name in PUBLIC_BINDINGS:
        if hasattr(extension, name):
            namespace[name] = getattr(extension, name)
