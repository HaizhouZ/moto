#ifndef MOTO_CONSTR_HPP
#define MOTO_CONSTR_HPP

#include <moto/ocp/func.hpp>
#include <moto/utils/tri_state.hpp>

namespace moto {
struct constr_impl; // fwd
class constr;
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
  protected:
    void value_impl(sp_approx_map &data) override;
    void jacobian_impl(sp_approx_map &data) override;
    void finalize_impl() override;
    struct type_hint {
        utils::tri_state_bool is_eq;
        bool is_soft;
    } field_hint_;
    friend class constr;

  public:
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
    sp_approx_map_ptr_t make_approx_map(sym_data &primal, approx_storage &raw, shared_data &shared) override {
        return constr_data_ptr_t(
            new constr_data(raw, std::move(*func_impl::make_approx_map(primal, raw, shared)), this));
    }
};
def_ptr(constr_impl);
/**
 * @brief wrapper of constr_impl, in fact a pointer
 *
 */
struct constr : public std::shared_ptr<constr_impl> {
    using impl_ptr_t = std::shared_ptr<constr_impl>;
    /**
     * @brief Construct a new constr object
     *
     * @param name name of the constraint
     * @param order approximation order
     * @param dim dimension of the constraint (0 for to be determined)
     * @param field field type, default to __undefined
     */
    constr(const std::string &name, approx_order order = approx_order::first, size_t dim = dim_tbd, field_t field = __undefined)
        : impl_ptr_t(new constr_impl(name, order, dim, field)) {}
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
        : impl_ptr_t(new constr_impl(name, order, out.size1(), field)) {
        assert(out.size2() == 1 && "constr output must be a column vector");
        (*this)->set_from_casadi(in_args, out);
    }
    /**
     * @brief set the constraint as equality constraint
     *
     * @param soft if true, set as soft equality constraint
     * @return constr& *this
     */
    constr &as_eq(bool soft = false) {
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
    constr &as_ineq() {
        (*this)->field_hint_.is_eq = false;
        return *this;
    }
    constr() = default;
    using impl_ptr_t::operator=;
    template <typename derived_impl>
        requires(std::derived_from<derived_impl, constr_impl>)
    /// @brief will get the shared ownership of impl_rval
    constr(derived_impl *impl_rval) : impl_ptr_t(impl_rval) {}
};
} // namespace moto

#endif // MOTO_CONSTR_HPP