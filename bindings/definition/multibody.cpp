#include <moto/multibody/casadi_manifold.hpp>
#include <type_cast.hpp>

void register_submodule_multibody(nb::module_ &m) {
    using namespace moto::multibody;
    nb::class_<casadi_manifold, moto::sym>(
        m, "casadi_manifold",
        "State symbol with user-supplied CasADi integration and difference maps for a nonlinear manifold.")
        .def_static("create", &casadi_manifold::create,
                    nb::arg("name"), nb::arg("q"), nb::arg("dq"),
                    nb::arg("integrated"), nb::arg("other"),
                    nb::arg("difference"), nb::arg("default_val") = nb::none(),
                    "Create paired current and terminal manifold states. ``integrated`` defines integrate(q, dq), while ``difference`` defines the tangent displacement from q to other");
}
