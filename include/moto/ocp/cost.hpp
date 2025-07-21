#include <moto/ocp/impl/func.hpp>

namespace moto {
/**
 * @brief simple cost implementation
 *
 */
class cost : public func_derived<cost> {
  protected:
    void finalize_impl() override;
    using base::base;

  public:
    struct finalize_hint {
        bool substitute_x_to_y = false; ///< whether to substitute x to y
    } hint_;
    static cost *create(const std::string &name, approx_order order = approx_order::second) {
        return base::create(name, order, 1, __cost);
    }

    static cost *create(const std::string &name, sym_init_list in_args, const cs::SX &out, approx_order order = approx_order::second) {
        assert(out.is_scalar() && "cost output must be a scalar");
        return base::create(name, in_args, out, order, __cost);
    }
    cost *as_terminal() {
        update_name(name_ + "_terminal");
        return this;
    }
    template <typename derived>
    requires std::is_base_of_v<cost, derived>
    derived* cast() {
        return base::moving_cast<derived, cost>(this);
    }
};
def_raw_ptr(cost);
} // namespace moto