#ifndef MOTO_CONSTR_IMPL_HPP
#define MOTO_CONSTR_IMPL_HPP

#include <moto/core/workspace_data.hpp>
#include <moto/ocp/impl/func.hpp>
#include <moto/solver/linesearch_config.hpp>
#include <moto/utils/optional_boolean.hpp>

namespace moto {
class constr;

/**
 * @brief constraint approximation with multipliers (and slack variables)
 */
class constr : public func {
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
         * @brief construct a new constr data object by moving from another sparse approximation map
         * @param multiplier reference to the multiplier vector
         * @param raw raw approximation storage
         * @param d sparse approximation map
         */
        approx_map(vector_ref multiplier, dense_approx_data &raw, func_approx_map &&d);
        /**
         * @brief construct a new constr data object, will bind multiplier to the raw data
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
        const func &f_;                ///< reference to the function
        approx_data(const func &f);
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

    template <typename derived = constr>
    using data_type = constr_data_tpl<typename derived::approx_map, typename derived::approx_data>;

  private:
    /// @brief type hint for the constraint
    struct field_hint {
        utils::optional_bool is_eq; ///< true if equality constraint, false if inequality constraint, default is unset
        bool is_soft = false;       ///< true if soft constraint, false if hard constraint, default is false
    } field_hint_;                  ///< type hint for the constraint

  protected:
    /// @brief evaluate the value of the constraint for merit
    void value_impl(func_approx_map &data) const override;
    /// @brief evaluate the jacobian of the constraint and the multiplier-jacobian product (vjp) for merit jacobian
    void jacobian_impl(func_approx_map &data) const override;
    /// @brief finalize the constraint, will be called upon added to a problem
    /// @note will set the field (if unset) based on the field hint and substitute __x to __y for pure-state constraints
    void finalize_impl() override;

  public:
    using base = func;
    using base::base; ///< inherit constructor from func
    void setup_workspace_data(func_arg_map &data, workspace_data *ws_data) const override {
        data.as<approx_map>().ls_cfg = &ws_data->as<solver::linesearch_config>();
    }
    SHARED_ATTR_GETTER(field_hint, constr); ///< getter for field hint
    /**
     * @brief make an approximation data for the constraint
     * @tparam derived derived type of @ref constr, default is constr
     * @param primal primal data
     * @param raw raw approximation data
     * @param shared shared data
     * @return data_type* pointer to the approximation data
     */
    template <typename derived = constr>
        requires(std::derived_from<derived, constr>)
    auto make_approx(sym_data &primal, dense_approx_data &raw, shared_data &shared) const {
        using derived_data_type = data_type<derived>;
        using map_base = constr::approx_map;
        using data_base = constr::approx_data;
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
     * @brief wrapped data maker for constr
     * @details if field_ is in @ref dense_approx_data::stored_constr_fields, it will return approx_map
     * otherwise it will call @ref make_approx to generate @ref constr::constr_data_tpl (with independent storage)
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
    /**
     * @brief set the constraint as equality constraint
     *
     * @param soft if true, set as soft equality constraint
     * @return constr& *this
     */
    // template <typename derived_impl = expr_type>
    //     requires(std::derived_from<derived_impl, expr_type>)
    // constr &as_eq(bool soft = false) {
    //     if (soft == true) {
    //         if (!std::derived_from<derived_impl, impl::soft_constr>) {
    //             throw std::runtime_error("as_eq(true) requires derived_impl to be soft_impl::expr_type");
    //         }
    //     }
    //     if constexpr (!std::is_same_v<derived_impl, expr_type>) {
    //         if (dynamic_cast<derived_impl *>(this->get()) == nullptr) { // check if the type is the same
    //             this->reset(new derived_impl(std::move(**this)));
    //         }
    //     }
    //     // setup hints
    //     if (soft)
    //         (*this)->field_hint_.is_soft = true;
    //     (*this)->field_hint_.is_eq = true;
    //     return *this;
    // }
    // /**
    //  * @brief set the constraint as inequality constraint
    //  *
    //  * @return constr& *this
    //  */
    // template <typename derived_impl>
    //     requires(std::derived_from<derived_impl, impl::ineq_constr>)
    // constr &as_ineq() {
    //     if (dynamic_cast<derived_impl *>(this->get()) == nullptr) { // check if the type is the same
    //         this->reset(new derived_impl(std::move(**this)));
    //     }
    //     (*this)->field_hint_.is_eq = false;
    //     return *this;
    // }
};
} // namespace moto

#endif // MOTO_CONSTR_IMPL_HPP