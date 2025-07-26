#include <moto/ocp/ineq_constr.hpp>
#include <moto/ocp/problem.hpp>

namespace moto {
void soft_constr::finalize_impl() {
    base::finalize_impl();
    fmt::println("good job!");
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

ineq_constr::approx_map::approx_map(dense_approx_data &raw, approx_map::map_base &&d)
    : base::approx_map(raw, std::move(d)),
      comp_(problem()->extract(raw.comp_[func_.field()], func_)) {}

void ineq_constr::value_impl(func_approx_map &data) const {
    base::value_impl(data);
    auto &d = data.as<approx_map>();
    d.comp_.array() = d.multiplier_.array() * d.v_.array();
} // namespace impl
} // namespace moto
