#include <moto/ocp/impl/func.hpp>

namespace moto {
/**
 * @brief simple cost implementation
 *
 */
class cost : public func {
  protected:
    void finalize_impl() override;
    using base = func; ///< inherit constructor from func
    using base::base;

  public:
    struct finalize_hint {
        bool substitute_x_to_y = false; ///< whether to substitute x to y
    } hint_;

    cost(const std::string &name, approx_order order = approx_order::second)
        : base(name, order, 1, __cost) {}

    cost(const std::string &name, sym_init_list in_args, const cs::SX &out, approx_order order = approx_order::second)
        : base(name, in_args, out, order, __cost) {
        assert(out.is_scalar() && "cost output must be a scalar");
    }
    cost &as_terminal() {
        name() += "_terminal";
        return *this;
    }
};

} // namespace moto