#include <moto/ocp/impl/ineq_constr.hpp>
#include <moto/ocp/impl/soft_constr.hpp>

namespace moto {
namespace impl {
void soft_constr::finalize_impl() {
    soft_constr_base::finalize_impl();
    if (!in_field(soft_constr_fields, field_))
        throw std::runtime_error(fmt::format(
            "Soft constraint {} must have field in soft_constr_fields, but got {}", name_, field::name(field_)));
}
void ineq_constr::finalize_impl() {
    impl::soft_constr_base::finalize_impl();
    if (!in_field(ineq_constr_fields, field_))
        throw std::runtime_error(fmt::format(
            "Inequality constraint {} must have field in ineq_constr_fields, but got {}", name_, field::name(field_)));
}
void ineq_constr::value_impl(sp_approx_map &data) {
    base::value_impl(data);
    auto &d = data.as<data_type::map_t>();
    d.comp_.array() = d.multiplier_.array() * d.v_.array();
}
} // namespace impl
} // namespace moto
