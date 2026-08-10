PUBLIC_BINDINGS = {
    "active_status_config",
    "approx_order",
    "constr",
    "cost",
    "casadi_manifold",
    "dense_dynamics",
    "semi_implicit_euler",
    "expr",
    "field",
    "func",
    "ineq",
    "lifted",
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
