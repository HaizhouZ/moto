#ifndef MOTO_CONSTR_IMPL_HPP
#define MOTO_CONSTR_IMPL_HPP

#include <moto/core/workspace_data.hpp>
#include <moto/ocp/impl/func.hpp>
#include <moto/solver/linesearch_config.hpp>
#include <moto/utils/optional_boolean.hpp>

namespace moto {
class generic_constr;
struct constr : public func {
    using func::func; ///< inherit base constructor
    constr(const std::string &name, approx_order order, size_t dim = dim_tbd, field_t field = __undefined);
    constr(const std::string &name, const var_inarg_list &in_args, const cs::SX &out,
           approx_order order = approx_order::first, field_t field = __undefined);
    generic_constr *operator->() const;
    /**
     * @brief set the constraint as soft equality constraint in-place
     * @return constr& *this
     */
    template <typename derived = generic_constr, typename base = generic_constr>
        requires(std::derived_from<derived, generic_constr> &&
                 std::derived_from<base, generic_constr>)
    constr &as_soft();
    /**
     * @brief set the constraint as inequality constraint in-place
     * @return constr& *this
     */
    template <typename derived = generic_constr, typename base = generic_constr>
        requires(std::derived_from<derived, generic_constr> &&
                 std::derived_from<base, generic_constr>)
    constr &as_ineq();
    // @brief set the constraint as inequality constraint with a specific type
    constr &as_ineq(std::string_view type_name);
};
/**
 * @brief constraint approximation with multipliers (and slack variables)
 */
class generic_constr : public generic_func {
  public:
    /**
     * @brief constraint approximation map
     * derived from func_approx_map with multipler and vjp (for cost) mapping in addition
     */
    struct approx_map : public func_approx_map {
        solver::linesearch_config *ls_cfg = nullptr; ///< line search configuration, can be nullptr
        double *merit_;                              ///< pointer to the merit value
        vector_ref multiplier_;                      ///< multiplier vector reference
        std::vector<row_vector_ref> vjp_;            ///< multiplier-jacobian product references (cost jacobian)
        /**
         * @brief construct a new generic_constr data object by moving from another sparse approximation map
         * @param multiplier reference to the multiplier vector
         * @param raw raw approximation storage
         * @param d sparse approximation map
         */
        approx_map(vector_ref multiplier, dense_approx_data &raw, func_approx_map &&d);
        /**
         * @brief construct a new generic_constr data object, will bind multiplier to the raw data
         * @param raw raw approximation storage
         * @param d sparse approximation map
         */
        approx_map(dense_approx_data &raw, func_approx_map &&d);
        /// @brief short-cut for nested moving construct
        approx_map(func_approx_map_ptr_t &&rhs) : approx_map(std::move(dynamic_cast<approx_map &>(*rhs))) {}

      protected:
        void map_merit_jac_from_raw(decltype(dense_approx_data::jac_) &raw, std::vector<row_vector_ref> &jac);
    };
    /**
     * @brief independent constraint approximation data
     * @note this is used to store the value and jacobian of the constraint
     */
    struct approx_data {
        vector v_data_;                ///< value data for the independent constraint
        std::vector<matrix> jac_data_; ///< jacobian data for the independent constraint
        const generic_func &f_;           ///< reference to the function
        approx_data(const generic_func &f);
    };

    /**
     * @brief constraint data, a pair of <map, data>
     *
     * @tparam approx_map_t mapping type, default is @ref approx_map
     * @tparam data_t data type, default is @ref approx_data
     */
    template <typename approx_map_t = approx_map,
              typename approx_data_t = approx_data>
        requires std::is_base_of_v<approx_map, approx_map_t> &&
                 std::is_base_of_v<approx_data, approx_data_t>
    struct constr_data_tpl final : public composed_data<approx_map_t, approx_data_t> {
        using base = composed_data<approx_map_t, approx_data_t>; ///< base type
        /// inherit base constructor
        using base::base;
        using map_type = approx_map_t;   ///< map type
        using data_type = approx_data_t; ///< data type
    };

  protected:
    /// @brief type hint for the constraint
    struct field_hint {
        utils::optional_bool is_eq = true; ///< true if equality constraint, false if inequality constraint, default is true
        bool is_soft = false;              ///< true if soft constraint, false if hard constraint, default is false
    } field_hint_;                         ///< type hint for the constraint

    /// @brief evaluate the value of the constraint for merit
    void value_impl(func_approx_map &data) const override;
    /// @brief evaluate the jacobian of the constraint and the multiplier-jacobian product (vjp) for merit jacobian
    void jacobian_impl(func_approx_map &data) const override;
    /// @brief finalize the constraint, will be called upon added to a problem
    /// @note will set the field (if unset) based on the field hint and substitute __x to __y for pure-state constraints
    void finalize_impl() override;

    friend class constr;

    using wrapper_type = constr;

  public:
    void setup_workspace_data(func_arg_map &data, workspace_data *ws_data) const override {
        data.as<approx_map>().ls_cfg = &ws_data->as<solver::linesearch_config>();
    }
    template <typename derived = generic_constr>
    using data_type = constr_data_tpl<typename derived::approx_map, typename derived::approx_data>;
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
    auto make_approx(sym_data &primal, dense_approx_data &raw, shared_data &shared) const {
        using derived_data_type = data_type<derived>;
        using map_base = generic_constr::approx_map;
        using data_base = generic_constr::approx_data;
        using map_derived = typename derived_data_type::map_type;
        using data_derived = typename derived_data_type::data_type;
        data_base d(*this);
        map_base base_map(raw, func_approx_map(primal, d.v_data_, to_matrix_ref_list(d.jac_data_), shared, *this));
        base_map.setup_hessian(raw);
        if constexpr (std::is_constructible_v<map_derived, dense_approx_data &, map_base &&>) {
            // if map_t can be constructed from dense_approx_data and func_approx_map
            map_derived derived_map(raw, std::move(base_map));
            return new derived_data_type(map_derived(std::move(derived_map)), data_derived(std::move(d)));
        } else {
            return new derived_data_type(map_derived(std::move(base_map)), data_derived(std::move(d)));
        }
    }
    /**
     * @brief wrapped data maker for generic_constr
     * @details if field_ is in @ref dense_approx_data::stored_constr_fields, it will return approx_map
     * otherwise it will call @ref make_approx to generate @ref generic_constr::constr_data_tpl (with independent storage)
     * @param primal primal data
     * @param raw approximation data
     * @param shared shared data
     * @return func_approx_map_ptr_t
     */
    func_approx_map_ptr_t create_approx_map(sym_data &primal, dense_approx_data &raw, shared_data &shared) const override {
        if (in_field(field(), dense_approx_data::stored_constr_fields))
            return func_approx_map_ptr_t(new approx_map(raw, func_approx_map(primal, raw, shared, *this)));
        else
            return func_approx_map_ptr_t(make_approx(primal, raw, shared));
    }
    DEF_FUNC_CLONE;
};

template <typename derived, typename base>
    requires(std::derived_from<derived, generic_constr>) && (std::derived_from<base, generic_constr>)
constr &constr::as_soft() {
    (*this)->field_hint_.is_soft = true;
    reset(new derived(std::move(static_cast<base &>(*this))));
    return *this;
}

template <typename derived, typename base>
    requires(std::derived_from<derived, generic_constr>) && (std::derived_from<base, generic_constr>)
constr &constr::as_ineq() {
    (*this)->field_hint_.is_eq = false;
    reset(new derived(std::move(static_cast<base &>(*this))));
    return *this;
}
} // namespace moto

#endif // MOTO_CONSTR_IMPL_HPP