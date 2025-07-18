#ifndef MOTO_OCP_IMPL_INEQ_CONSTR_HPP
#define MOTO_OCP_IMPL_INEQ_CONSTR_HPP

#include <moto/ocp/impl/soft_constr.hpp>

namespace moto {
namespace impl {
/**
 * @brief inequality constraint approximation map with the complementarity residual map
 *
 */
struct ineq_constr_approx_map : public soft_constr_approx_map {
    vector_ref comp_;
    ineq_constr_approx_map(approx_storage &raw, constr_approx_map &&d)
        : soft_constr_approx_map(raw, std::move(d)),
          comp_(problem()->extract(raw.comp_[func_.field_], func_)) {}
};
/**
 * @brief inequality constraint interface class
 *
 */
class ineq_constr : public soft_constr {
  private:
    using base = soft_constr;

  protected:
    /// @brief data_type for the inequality constraint
    using data_type = constr_data<ineq_constr_approx_map, base::data_type::data_t>;
    /// @brief check if the field is in the soft constraint fields
    void finalize_impl() override;
    /// @brief evaluate the value of the constraint and compute the complementarity residual
    /// @param d data
    void value_impl(sp_approx_map& d) override;

  public:
    using base::base;
    /***
     * @brief make approximation data for the inequality constraint, will use default @ref data_type
     */
    sp_approx_map_ptr_t make_approx_map(sym_data &primal, approx_storage &raw, shared_data &shared) override {
        return sp_approx_map_ptr_t(make_approx<data_type>(primal, raw, shared));
    }
};

} // namespace impl
} // namespace moto

#endif // MOTO_OCP_IMPL_INEQ_CONSTR_HPP