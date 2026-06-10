#ifndef MOTO_OCP_COST_HPP
#define MOTO_OCP_COST_HPP

#include <moto/ocp/impl/func.hpp>

namespace moto {
class generic_cost;
using cost = utils::shared<generic_cost>;
/**
 * @brief simple cost implementation
 *
 */
class generic_cost : public generic_func {
  protected:
    bool use_gauss_newton_ = false;

    void finalize_impl() override;
    var gn_weight_; ///< weight for gauss-newton cost

  public:
    using base = generic_func;
    using base::base; ///< inherit base constructor

    generic_cost(const std::string &name, approx_order order = approx_order::second);
    generic_cost(const std::string &name, const var_inarg_list &in_args, const cs::SX &out,
                 approx_order order = approx_order::second);

    DEF_DEFAULT_CLONE(generic_cost)

    generic_cost *set_diag_hess();                     ///< set hessian to be diagonal
    generic_cost *set_gauss_newton(const var &weight); ///< set to convex-over-nonlinear cost
};

} // namespace moto

#endif // MOTO_OCP_COST_HPP
