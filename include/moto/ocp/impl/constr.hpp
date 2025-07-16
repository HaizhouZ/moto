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
struct constr_approx_map : public sp_approx_map {
    /// @todo: add this to raw
    // vector_ref slack_;
    double *merit_;
    vector_ref multiplier_;
    std::vector<row_vector_ref> vjp_;
    /**
     * @brief construct a new constr data object by moving from another sparse approximation map
     * @param multiplier reference to the multiplier vector
     * @param raw raw approximation storage
     * @param d sparse approximation map
     */
    constr_approx_map(vector_ref multiplier, approx_storage &raw, sp_approx_map &&d);
    /**
     * @brief construct a new constr data object, will bind multiplier to the raw data
     * @param raw raw approximation storage
     * @param d sparse approximation map
     */
    constr_approx_map(approx_storage &raw, sp_approx_map &&d);
    /// @brief short-cut for nested moving construct
    constr_approx_map(sp_approx_map_ptr_t &&rhs) : constr_approx_map(std::move(dynamic_cast<constr_approx_map &>(*rhs))) {}
};

struct constr_approx_data {
    vector v_data_;                ///< value data for the independent constraint
    std::vector<matrix> jac_data_; ///< jacobian data for the independent constraint
    func *f_;                      ///< reference to the function
    constr_approx_data(func &f);
};

/**
 * @brief constraint approximation with multipliers (and slack variables)
 */
class constr : public func {
  private:
    struct constr_data_base {}; ///< base class for constr data as constraint

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
     * @brief constraint data
     *
     * @tparam approx_map_t mapping type, default is constr_approx_map
     * @tparam data_t data type, default is constr_approx_data
     */
    template <typename approx_map_t = constr_approx_map,
              typename data_t = constr_approx_data>
        requires std::is_base_of_v<constr_approx_map, approx_map_t> && std::is_base_of_v<constr_approx_data, data_t>
    struct constr_data final : private constr_data_base, public composed_data<approx_map_t, data_t> {
        using base = composed_data<approx_map_t, data_t>; ///< base type
        /// inherit base constructor
        using base::base;
        using mtype = approx_map_t; ///< map type
        using dtype = data_t;       ///< data type
    };
    /**
     * @brief make an approximation data for the constraint
     *
     * @tparam data_type type of the data, default is constr_data<>
     * @param primal primal data
     * @param raw raw approximation data
     * @param shared shared data
     * @return data_type* pointer to the approximation data
     */
    template <typename data_type = constr_data<>>
        requires std::is_base_of_v<constr_data_base, data_type>
    auto make_approx(sym_data &primal, approx_storage &raw, shared_data &shared) {
        using map_t = typename data_type::mtype;
        using data_t = typename data_type::dtype;
        constr_approx_data d(*this);
        auto base_map = constr_approx_map(raw, sp_approx_map(primal, d.v_data_, to_matrix_ref_list(d.jac_data_), shared, *this));
        base_map.setup_hessian(raw);
        return new data_type(map_t(std::move(base_map)), data_t(std::move(d)));
    }
    /**
     * @brief wrapped data maker for constr
     * @details if field approx is in @ref approx_storage::stored_constr_fields, it will return constr_approx_map
     * otherwise it will call @ref make_approx to generate @ref constr_data (with independent storage)
     * @param primal primal data
     * @param raw approximation data
     * @param shared shared data
     * @return sp_approx_map_ptr_t
     */
    sp_approx_map_ptr_t make_approx_map(sym_data &primal, approx_storage &raw, shared_data &shared) override {
        if (in_field(approx_storage::stored_constr_fields, field_))
            return sp_approx_map_ptr_t(new constr_approx_map(raw, sp_approx_map(primal, raw, shared, *this)));
        else
            return sp_approx_map_ptr_t(make_approx(primal, raw, shared));
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