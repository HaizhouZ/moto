#ifndef MOTO_OCP_IMPL_INEQ_CONSTR_HPP
#define MOTO_OCP_IMPL_INEQ_CONSTR_HPP

#include <moto/ocp/soft_constr.hpp>

namespace moto {
/**
 * @brief inequality constraint interface class
 *
 */
class ineq_constr : public soft_constr {
  private:
    using base = soft_constr;

  public:
    /**
     * @brief inequality constraint approximation map with the complementarity residual map
     *
     */
    struct approx_data : public base::approx_data {
        vector_ref comp_;
        approx_data(data_base &&d);
    };

  protected:
    /// @brief finalize the inequality constraint, will be called upon added to a problem
    void finalize_impl() override;
    /// @brief evaluate the value of the constraint and compute the complementarity residual
    void value_impl(func_approx_data &data) const override;

  public:
    using base::base;
    ineq_constr(generic_constr &&rhs) : base(std::move(rhs)) {
        field_hint().is_eq = false; ///< set the field hint to inequality
    } ///< move constructor from generic_constr
    /***
     * @brief make approximation data for the inequality constraint, will use default @ref data_type
     */
    func_approx_data_ptr_t create_approx_data(sym_data &primal, lag_data &raw, shared_data &shared) const override {
        return func_approx_data_ptr_t(make_approx<ineq_constr>(primal, raw, shared));
    }
};

} // namespace moto

#endif // MOTO_OCP_IMPL_INEQ_CONSTR_HPP
