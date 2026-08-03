PUBLIC_BINDINGS = {
    "active_status_config",
    "approx_order",
    "constr",
    "cost",
    "casadi_manifold",
    "dense_dynamics",
    "sparse_dynamics",
    "expr",
    "field",
    "func",
    "ineq",
    "endpoint",
    "pmm_constr",
    "quaternion",
    "stage_ocp",
    "sym",
}


def export_public_bindings(extension, namespace):
    for name in PUBLIC_BINDINGS:
        if hasattr(extension, name):
            namespace[name] = getattr(extension, name)
    namespace["stage"] = extension.stage_ocp.create
