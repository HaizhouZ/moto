#ifndef MOTO_CONSTR_IMPL_HPP
#define MOTO_CONSTR_IMPL_HPP

#include <moto/core/workspace_data.hpp>
#include <moto/ocp/impl/func.hpp>
#include <moto/solver/linesearch_config.hpp>
#include <moto/utils/optional_boolean.hpp>

namespace moto {
class generic_constr;                         ///< forward declaration
using constr = utils::shared<generic_constr>; ///< generic constr holder
/**
 * @brief constraint approximation with multipliers (and slack variables)
 */
class generic_constr : public generic_func {
  public:
    /**
     * @brief constraint approximation map
     * derived from func_approx_data with multipler and vjp (for cost) mapping in addition
     */
    struct approx_data : public func_approx_data {
        solver::linesearch_config *ls_cfg = nullptr; ///< line search configuration, can be nullptr
        scalar_t *merit_;                            ///< pointer to the merit value
        vector_ref multiplier_;                      ///< multiplier vector reference
        /**
         * @brief construct a new generic_constr data object by moving from another sparse approximation map
         * @param multiplier reference to the multiplier vector
         * @param raw raw approximation storage
         * @param d sparse approximation map
         */
        approx_data(vector_ref multiplier, merit_data &raw, func_approx_data &&d);
        /**
         * @brief construct a new generic_constr data object, will bind multiplier to the raw data
         * @param raw raw approximation storage
         * @param d sparse approximation map
         */
        approx_data(func_approx_data &&d);

      protected:
        void map_merit_jac_from_raw(decltype(merit_data::merit_jac_) &raw, std::vector<row_vector_ref> &jac);
    };

  protected:
    /// @brief type hint for the constraint
    struct field_hint {
        utils::optional_bool is_eq = true; ///< true if equality constraint, false if inequality constraint, default is true
        bool is_soft = false;              ///< true if soft constraint, false if hard constraint, default is false
    } field_hint_;                         ///< type hint for the constraint

    /// @brief finalize the constraint, will be called upon added to a problem
    /// @note will set the field (if unset) based on the field hint and substitute __x to __y for pure-state constraints
    void finalize_impl() override;

  public:
    void setup_workspace_data(func_arg_map &data, workspace_data *ws_data) const override {
        data.as<approx_data>().ls_cfg = &ws_data->as<solver::linesearch_config>();
    }
    template <typename derived = generic_constr>
    using data_type = derived::approx_data; // constr_data_tpl<typename derived::approx_data, typename derived::approx_data>;
    using base = generic_func;
    using base::base; ///< inherit base constructor

    PROPERTY(field_hint); ///< getter for field hint

    /**
     * @brief make an approximation data for the constraint
     * @tparam derived derived type of @ref generic_constr, default is generic_constr
     * @param primal primal data
     * @param raw raw approximation data
     * @param shared shared data
     * @return data_type* pointer to the approximation data
     */
    template <typename derived = generic_constr>
        requires(std::derived_from<derived, generic_constr>)
    auto make_approx(sym_data &primal, merit_data &raw, shared_data &shared) const {
        using data_base = generic_constr::approx_data;
        using data_derived = typename derived::approx_data;
        data_base d(func_approx_data(primal, raw, shared, *this));
        return new data_derived(std::move(d));
    }
#define OVERLOAD_CREATE_APPROX_DATA(derived)                                                                           \
    func_approx_data_ptr_t create_approx_data(sym_data &primal, merit_data &raw, shared_data &shared) const override { \
        return func_approx_data_ptr_t(make_approx<derived>(primal, raw, shared));                                      \
    }
    /**
     * @brief wrapped data maker for generic_constr
     * @details if field_ is in @ref merit_data::stored_constr_fields, it will return approx_data
     * otherwise it will call @ref make_approx to generate @ref generic_constr::constr_data_tpl (with independent storage)
     * @param primal primal data
     * @param raw approximation data
     * @param shared shared data
     * @return func_approx_data_ptr_t
     */
    OVERLOAD_CREATE_APPROX_DATA(generic_constr);
    DEF_DEFAULT_CLONE(generic_constr);

    // @brief make an inequality constraint from this constraint by moving
    generic_constr *cast_ineq(std::string_view type_name);
    // @brief make a soft equality constraint from this constraint by moving
    generic_constr *cast_soft(std::string_view type_name);
};
} // namespace moto

#endif // MOTO_CONSTR_IMPL_HPP