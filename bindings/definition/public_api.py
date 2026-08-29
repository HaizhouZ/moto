PUBLIC_BINDINGS = {
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
    "pmm_constr",
    "quaternion",
    "sym",
}


def export_public_bindings(extension, namespace):
    for name in PUBLIC_BINDINGS:
        if hasattr(extension, name):
            namespace[name] = getattr(extension, name)

    def stage():
        """Create an authored OCP stage."""
        return extension.stage_ocp.create()

    stage.__module__ = "moto"
    namespace["stage"] = stage
