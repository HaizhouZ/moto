#include <moto/ocp/constr.hpp>
#include <moto/ocp/sym.hpp>
#include <moto/ocp/usr_func.hpp>
#include <nanobind/stl/function.h>
#include <nanobind/stl/variant.h>
#include <type_cast.hpp>

#include <moto/multibody/quaternion.hpp>

#include <moto/ocp/dynamics/dense_dynamics.hpp>

void register_submodule_multibody(nb::module_ &m) {
    using namespace moto::multibody;
    nb::class_<quaternion>(m, "quaternion", "quaternion symbolic variable, representing a unit quaternion")
        .def_static("multiply", &quaternion::multiply, nb::arg("q0"), nb::arg("q1"), "quaternion multiplication q0 * q1")
        .def_static("inverse", &quaternion::inverse, nb::arg("q"), "quaternion inverse q^{-1}")
        .def_static("identity", &quaternion::symbolic_identity, "identity quaternion [0, 0, 0, 1]")
        .def_static("exp3", &quaternion::exp3, nb::arg("w"), nb::arg("tolerance") = 1e-12, "exponential map from R^3 to unit quaternion")
        .def_static("log3", &quaternion::log3, nb::arg("q"), nb::arg("tolerance") = 1e-12, "logarithm map from unit quaternion to R^3")
        .def_static("create", &quaternion::create, nb::arg("name"), "create a new quaternion symbolic variable");
}
