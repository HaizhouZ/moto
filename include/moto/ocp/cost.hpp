#include <moto/ocp/impl/func.hpp>

namespace moto {
/**
 * @brief simple cost implementation
 *
 */
class cost : public func {
  protected:
    void finalize_impl() override;

  public:
    struct impl : public func::impl {
        struct finalize_hint {
            bool substitute_x_to_y = false; ///< whether to substitute x to y
        } finalize_hint_;

        using func::impl::impl; ///< inherit constructor from func::impl
        impl(func::impl &&rhs) : func::impl(std::move(rhs)) {}
    };

    DEF_PROTECTED_SHARED_GETTER();

  public:
    using base = func; ///< inherit constructor from func

    SHARED_ATTR_GETTER(finalize_hint, cost); ///< getter for finalize_hint

    cost(const std::string &name, approx_order order = approx_order::second)
        : base(name, order, 1, __cost) {
        shared_ = std::make_shared<impl>(std::move(*shared_));
    }

    cost(const std::string &name, sym_init_list in_args, const cs::SX &out, approx_order order = approx_order::second)
        : base(name, in_args, out, order, __cost) {
        assert(out.is_scalar() && "cost output must be a scalar");
        shared_.reset(new impl(std::move(static_cast<base::impl&>(*shared_))));
    }
    cost &as_terminal() {
        name() += "_terminal";
        return *this;
    }
};

} // namespace moto