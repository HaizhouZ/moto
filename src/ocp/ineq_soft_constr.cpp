#include <moto/ocp/impl/ineq_constr.hpp>
#include <moto/ocp/impl/soft_constr.hpp>

namespace moto {
namespace impl {
void soft_constr::finalize_impl() {
    base::finalize_impl();
    if (!skip_field_check && !in_field(field_, soft_constr_fields))
        throw std::runtime_error(fmt::format(
            "Soft constraint {} must have field in soft_constr_fields, but got {}", name_, field::name(field_)));
}
void ineq_constr::finalize_impl() {
    skip_field_check = true;
    base::finalize_impl();
    if (!in_field(field_, ineq_constr_fields))
        throw std::runtime_error(fmt::format(
            "Inequality constraint {} must have field in ineq_constr_fields, but got {}", name_, field::name(field_)));
}
void ineq_constr::value_impl(func_approx_map &data) {
    base::value_impl(data);
    auto &d = data.as<approx_map>();
    d.comp_.array() = d.multiplier_.array() * d.v_.array();
}
} // namespace impl
} // namespace moto
