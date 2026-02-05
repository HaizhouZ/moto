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
    struct finalize_hint {
        bool substitute_x_to_y = false;
        bool gauss_newton = false;
    } finalize_hint_;

    void finalize_impl() override;
    var gn_weight_; ///< weight for gauss-newton cost
    friend struct cost;
    using wrapper_type = cost;

    
    public:
    using base = generic_func;
    using base::base; ///< inherit base constructor
    
    generic_cost(const std::string &name, approx_order order = approx_order::second);
    generic_cost(const std::string &name, const var_inarg_list &in_args, const cs::SX &out,
                 approx_order order = approx_order::second);
                 
    PROPERTY(finalize_hint)

    DEF_DEFAULT_CLONE(generic_cost)


    generic_cost* set_diag_hess();
    generic_cost* as_terminal();                       ///< convert to terminal cost
    generic_cost* set_gauss_newton(const var &weight); ///< convert to convex-over-nonlinear cost
};

} // namespace moto