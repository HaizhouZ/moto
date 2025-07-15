#ifndef MOTO_CONSTR_HPP
#define MOTO_CONSTR_HPP

#include <moto/ocp/impl/constr.hpp>

namespace moto {
/**
 * @brief pointer wrapper of expr_type
 *
 */
struct constr : public impl::shared_<impl::constr, constr> {
    /**
     * @brief Construct a new constr object
     *
     * @param name name of the constraint
     * @param order approximation order
     * @param dim dimension of the constraint (0 for to be determined)
     * @param field field type, default to __undefined
     */
    constr(const std::string &name, approx_order order = approx_order::first, size_t dim = dim_tbd, field_t field = __undefined)
        : shared_(new expr_type(name, order, dim, field)) {}
    /**
     * @brief Construct a new constr object from casadi SX expression
     *
     * @param name  name of the constraint
     * @param in_args  input arguments
     * @param out output casadi SX expression
     * @param order approximation order
     * @param field field type, default to __undefined
     */
    constr(const std::string &name, std::initializer_list<sym> in_args, const cs::SX &out,
           approx_order order = approx_order::first, field_t field = __undefined)
        : shared_(new expr_type(name, order, out.size1(), field)) {
        assert(out.size2() == 1 && "constr output must be a column vector");
        (*this)->set_from_casadi(in_args, out);
    }
    template <typename derived_impl>
        requires(std::derived_from<derived_impl, expr_type>)
    /// @brief will get the shared ownership of impl_rval
    constr(derived_impl *impl_rval) : shared_(impl_rval) {}
    /**
     * @brief set the constraint as equality constraint
     *
     * @param soft if true, set as soft equality constraint
     * @return constr& *this
     */
    template <typename derived_impl = expr_type>
        requires(std::derived_from<derived_impl, expr_type>)
    constr &as_eq(bool soft = false) {
        if (soft == true) {
            if (!std::derived_from<derived_impl, impl::soft_constr_base>) {
                throw std::runtime_error("as_eq(true) requires derived_impl to be soft_impl::expr_type");
            }
        }
        if constexpr (!std::is_same_v<derived_impl, expr_type>) {
            if (dynamic_cast<derived_impl *>(this->get()) == nullptr) { // check if the type is the same
                this->reset(new derived_impl(std::move(**this)));
            }
        }
        // setup hints
        if (soft)
            (*this)->field_hint_.is_soft = true;
        (*this)->field_hint_.is_eq = true;
        return *this;
    }
    /**
     * @brief set the constraint as inequality constraint
     *
     * @return constr& *this
     */
    template <typename derived_impl>
        requires(std::derived_from<derived_impl, impl::soft_constr_base>)
    constr &as_ineq() {
        if (dynamic_cast<derived_impl *>(this->get()) == nullptr) { // check if the type is the same
            this->reset(new derived_impl(std::move(**this)));
        }
        (*this)->field_hint_.is_eq = false;
        return *this;
    }
    constr &as_eq(std::string_view type_name, bool soft = false);
    constr &as_ineq(std::string_view type_name);
    constr() = default;
    using shared_::operator=;
};
} // namespace moto

#endif // MOTO_CONSTR_HPP