#ifndef MOTO_OCP_COST_HPP
#define MOTO_OCP_COST_HPP

#include <moto/ocp/impl/func.hpp>
#include <moto/ocp/sym.hpp>

namespace moto {
class generic_cost;
using cost = utils::shared<generic_cost>;
/**
 * @brief simple cost implementation
 *
 */
class generic_cost : public generic_func {
  public:
    using tracking_param = std::variant<scalar_t, vector, var>;

  protected:
    bool use_gauss_newton_ = false;

    void finalize_impl() override;
    void substitute(const sym &arg, const sym &rhs) override;
    var gn_weight_; ///< weight for gauss-newton cost
    var weight_;    ///< parameter created by the tracking-cost factories
    var reference_; ///< parameter created by the tracking-cost factories
    static cost make_tracking(const std::string &name,
                              const var_inarg_list &args,
                              const cs::SX &value,
                              tracking_param weight,
                              tracking_param reference);

  public:
    using base = generic_func;
    using base::base; ///< inherit base constructor

    generic_cost(const std::string &name, approx_order order = approx_order::second);
    generic_cost(const std::string &name, const var_inarg_list &in_args, const cs::SX &out,
                 approx_order order = approx_order::second);

    const var &weight() const { return weight_; }
    const var &reference() const { return reference_; }

    static cost from_scalar(const std::string &name,
                            const var_inarg_list &args,
                            const cs::SX &value,
                            tracking_param weight = scalar_t(1),
                            tracking_param reference = scalar_t(0));
    static cost from_vector(const std::string &name,
                            const var_inarg_list &args,
                            const cs::SX &value,
                            tracking_param weight = scalar_t(1),
                            tracking_param reference = scalar_t(0));

  protected:
    DEF_DEFAULT_CLONE(generic_cost)
};

} // namespace moto

#endif // MOTO_OCP_COST_HPP
