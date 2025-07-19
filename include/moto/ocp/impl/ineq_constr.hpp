#ifndef MOTO_OCP_IMPL_INEQ_CONSTR_HPP
#define MOTO_OCP_IMPL_INEQ_CONSTR_HPP

#include <moto/ocp/impl/soft_constr.hpp>

namespace moto {
namespace impl {
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
    struct approx_map : public base::approx_map {
        vector_ref comp_;
        template <typename approx_map_t>
        approx_map(dense_approx_data &raw, approx_map_t &&d)
            : base::approx_map(raw, std::move(d)),
              comp_(problem()->extract(raw.comp_[func_.field_], func_)) {}
    };

  protected:
    /// @brief check if the field is in the soft constraint fields
    void finalize_impl() override;
    /// @brief evaluate the value of the constraint and compute the complementarity residual
    /// @param d data
    void value_impl(func_approx_map &d) override;

  public:
    using base::base;
    /***
     * @brief make approximation data for the inequality constraint, will use default @ref data_type
     */
    func_approx_map_ptr_t create_approx_map(sym_data &primal, dense_approx_data &raw, shared_data &shared) override {
        return func_approx_map_ptr_t(make_approx<ineq_constr>(primal, raw, shared));
    }
};

} // namespace impl
} // namespace moto

#endif // MOTO_OCP_IMPL_INEQ_CONSTR_HPP