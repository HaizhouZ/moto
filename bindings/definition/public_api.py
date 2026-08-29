PUBLIC_BINDINGS = {
    "approx_order": "approx_order",
    "casadi_manifold": "casadi_manifold",
    "constr": "constr",
    "cost": "cost",
    "dense_dynamics": "dense_dynamics",
    "endpoint": "endpoint",
    "expr": "expr",
    "field": "field",
    "func": "func",
    "ineq": "ineq",
    "lifted": "lifted",
    "pmm_constr": "pmm_constr",
    "semi_implicit_euler": "semi_implicit_euler",
    "sqp": "sqp",
    "stage_ocp": "stage_ocp",
    "sym": "sym",
}


def publish_type(cls, public_name):
    source_module = cls.__module__
    visited = set()

    def publish_function(function, qualname):
        if getattr(function, "__module__", None) == source_module:
            function.__module__ = "moto"
            function.__qualname__ = qualname

    def publish(current, qualname):
        if id(current) in visited:
            return
        visited.add(id(current))
        current.__module__ = "moto"
        current.__name__ = qualname.rsplit(".", 1)[-1]
        current.__qualname__ = qualname
        if current.__doc__ is None:
            current.__doc__ = f"Public API type ``moto.{qualname}``."
        for name, child in vars(current).items():
            if isinstance(child, type) and child.__module__ == source_module:
                publish(child, f"{qualname}.{name}")
            elif isinstance(child, (staticmethod, classmethod)):
                publish_function(child.__func__, f"{qualname}.{name}")
            elif isinstance(child, property):
                for function in (child.fget, child.fset, child.fdel):
                    if function is not None:
                        publish_function(function, f"{qualname}.{name}")
            else:
                publish_function(child, f"{qualname}.{name}")

    publish(cls, public_name)


def export_public_bindings(extension, namespace):
    for public_name, native_name in PUBLIC_BINDINGS.items():
        if hasattr(extension, native_name):
            binding = getattr(extension, native_name)
            if isinstance(binding, type):
                publish_type(binding, public_name)
            namespace[public_name] = binding

    def stage():
        """Create an authored OCP stage."""
        return extension.stage_ocp.create()

    stage.__module__ = "moto"
    stage.__annotations__["return"] = extension.stage_ocp
    namespace["stage"] = stage
