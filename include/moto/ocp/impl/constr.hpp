#ifndef MOTO_CONSTR_IMPL_HPP
#define MOTO_CONSTR_IMPL_HPP

#include <moto/ocp/impl/func.hpp>
#include <moto/utils/optional_boolean.hpp>

namespace moto {
namespace impl {
struct constr;
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
    /**
     * @brief construct a new constr data object by moving from another sparse approximation map
     *
     * @param raw raw approximation storage
     * @param d sparse approximation map
     * @param cstr pointer to the constr object
     */
    constr_data(approx_storage &raw, sp_approx_map &&d, constr *cstr);
    /// @brief short-cut for nested moving construct
    constr_data(sp_approx_map_ptr_t &&rhs) : constr_data(std::move(dynamic_cast<constr_data &>(*rhs))) {}
};
/**
 * @brief constraint approximation with multipliers (and slack variables)
 */
class constr : public func {
  protected:
    /// @brief evaluate the value of the constraint for merit
    void value_impl(sp_approx_map &data) override;
    /// @brief evaluate the jacobian of the constraint and the multiplier-jacobian product (vjp) for merit jacobian
    void jacobian_impl(sp_approx_map &data) override;
    /// @brief finalize the constraint, will be called upon added to a problem
    /// @note will set the field (if unset) based on the field hint and substitute __x to __y for pure-state constraints
    void finalize_impl() override;

  public:
    /**
     * @brief type hint for the constraint
     *
     */
    struct field_hint {
        utils::optional_bool is_eq; ///< true if equality constraint, false if inequality constraint, default is unset
        bool is_soft = false;       ///< true if soft constraint, false if hard constraint, default is false
    } field_hint_;                  ///< type hint for the constraint

    /**
     * @brief construct a new constraint
     *
     * @param name  name of the constraint
     * @param order approximation order, default is first order
     * @param dim   dimension of the constraint, default is 0 (to be determined)
     * @param field field type, default is __undefined
     */
    constr(const std::string &name, approx_order order = approx_order::first,
           size_t dim = dim_tbd, field_t field = __undefined)
        : func(name, order, dim, field) {
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
        return std::make_unique<constr_data>(raw, std::move(*func::make_approx_map(primal, raw, shared)), this);
    }
};
/**
 * @brief soft constraint abstract base class
 *
 */
struct soft_constr_base : public constr {
    /**
     * @brief move constructor from constr
     *
     * @param rhs constr to move from
     */
    soft_constr_base(constr &&rhs) : constr(std::move(rhs)) {}
};
} // namespace impl
} // namespace moto

#endif // MOTO_CONSTR_IMPL_HPP