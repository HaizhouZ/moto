#ifndef MOTO_CONSTR_HPP
#define MOTO_CONSTR_HPP

#include <moto/ocp/func.hpp>
#include <moto/utils/tri_state.hpp>

namespace moto {
struct constr_impl; // fwd
/**
 * @brief constraint data
 * derived from sp_approx_map with multipler and vjp (for cost) mapping in addition
 */
struct constr_data : public sp_approx_map {
    /// @todo: add this to raw
    // vector_ref slack_;
    double *merit_;
    vector_ref multiplier_;
    std::vector<row_vector_ref> vjp_;
    constr_data(approx_storage &raw, sp_approx_map &&d, constr_impl *cstr);
};
def_unique_ptr(constr_data);
/**
 * @brief constraint approximation with multipliers (and slack variables)
 */
class constr_impl : public func_impl {
  private:
    void value_impl(sp_approx_map &data) override final;
    void jacobian_impl(sp_approx_map &data) override final;
    bool finalize_impl() override;
    struct field_hint_t {
        utils::tri_state_bool is_eq;
        utils::tri_state_bool is_soft;
    } field_hint_;

  public:
    /// @brief constructor used for type deduction
    constr_impl(const std::string &name, approx_order order = approx_order::first,
                size_t dim = dim_tbd, utils::tri_state_bool _is_eq = utils::Unset, utils::tri_state_bool _is_soft = utils::Unset)
        : func_impl(name, order, dim, __undefined), field_hint_((field_hint_t){.is_eq = _is_eq, .is_soft = _is_soft}) {
    }
    /// @brief constructor with explicit field type
    constr_impl(const std::string &name, approx_order order = approx_order::first,
                size_t dim = dim_tbd, field_t field = __undefined)
        : func_impl(name, order, dim, field) {
    }

    /**
     * @brief wrapped data maker for constr
     *
     * @param primal ptr to primal data
     * @param raw ptr to approximation data
     * @param shared ptr to shared data
     * @return sp_approx_map_ptr_t
     */
    sp_approx_map_ptr_t make_approx_data_mapping(sym_data &primal, approx_storage &raw, shared_data &shared) override {
        return constr_data_ptr_t(
            new constr_data(raw, std::move(*func_impl::make_approx_data_mapping(primal, raw, shared)), this));
    }
};
def_ptr(constr_impl);
/**
 * @brief wrapper of constr_impl, in fact a pointer
 *
 */
struct constr : public std::shared_ptr<constr_impl> {
    using impl_ptr_t = std::shared_ptr<constr_impl>;
    constr(const std::string &name, approx_order order = approx_order::first, size_t dim = dim_tbd, field_t field = __undefined)
        : impl_ptr_t(new constr_impl(name, order, dim, field)) {}
    constr() = default;
    using impl_ptr_t::operator=;
    template <typename derived_impl>
        requires(std::derived_from<derived_impl, constr_impl>)
    /// @brief will get the shared ownership of impl_rval
    constr(derived_impl *impl_rval) : impl_ptr_t(impl_rval) {}
};

template <typename impl_t>
    requires(std::derived_from<impl_t, constr_impl>)
struct eq_constr_tpl : public std::shared_ptr<impl_t> {
    using impl_ptr_t = std::shared_ptr<impl_t>;
    eq_constr_tpl(const std::string &name, approx_order order = approx_order::first, size_t dim = dim_tbd, bool soft = false)
        : impl_ptr_t(new impl_t(name, order, true, dim, soft)) {}
    eq_constr_tpl() = default;
    using impl_ptr_t::operator=;
    eq_constr_tpl(impl_t *impl_rval) : impl_ptr_t(impl_rval) {} ///< will get the ownership of impl_rval
};

typedef eq_constr_tpl<constr_impl> eq_constr; ///< instantiate eq_constr_impl_wrapper

template <typename impl_t>
    requires(std::derived_from<impl_t, constr_impl>)
struct ineq_constr_tpl : public std::shared_ptr<impl_t> {
    using impl_ptr_t = std::shared_ptr<impl_t>;
    ineq_constr_tpl(const std::string &name, approx_order order = approx_order::first, size_t dim = dim_tbd)
        : impl_ptr_t(new impl_t(name, order, false, dim)) {}
    ineq_constr_tpl() = default;
    using impl_ptr_t::operator=;
    ineq_constr_tpl(impl_t *impl_rval) : impl_ptr_t(impl_rval) {} ///< will get the ownership of impl_rval
};

typedef ineq_constr_tpl<constr_impl> ineq_constr; ///< instantiate ineq_constr_impl_wrapper
} // namespace moto

#endif // MOTO_CONSTR_HPP