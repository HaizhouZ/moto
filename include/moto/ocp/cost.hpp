#include <moto/ocp/impl/func.hpp>

namespace moto {
/**
 * @brief simple cost implementation
 *
 */
class cost : public func_base {
  protected:
    struct finalize_hint {
        bool substitute_x_to_y = false;
    } finalize_hint_;

    void finalize_impl() override;

  public:
    using base = func_base;
    cost() = default;

    PROPERTY(finalize_hint)

    cost(const std::string &name, approx_order order = approx_order::second)
        : base(name, order, 1, __cost) {}

    cost(const std::string &name, const var_list &in_args, const cs::SX &out, approx_order order = approx_order::second)
        : base(name, in_args, out, order, __cost) {
        assert(out.is_scalar() && "cost output must be a scalar");
    }
    cost as_terminal() {
        cost tmp(*this);
        tmp.name() += "_terminal";
        return tmp;
    }
};

} // namespace moto