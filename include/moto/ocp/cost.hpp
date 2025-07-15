#include <moto/ocp/impl/func.hpp>

namespace moto {
struct cost; // forward declaration
namespace impl {
/**
 * @brief simple cost implementation
 *
 */
class cost : public func {
  protected:
    void substitute_x_to_y();
    friend struct moto::cost;

  public:
    cost(const std::string &name, approx_order order = approx_order::second)
        : func(name, order, 1, __cost) {}
};
} // namespace impl
/**
 * @brief wrapper of impl::cost, in fact a pointer
 *
 */
struct cost : public impl::shared_<impl::cost, cost> {
    /**
     * @brief Construct a new cost object
     *
     * @param name name of the cost
     */
    cost(const std::string &name)
        : shared_(new expr_type(name)) {
    }
    /**
     * @brief Construct a new cost object from casadi expression
     *
     * @param name name of the cost
     * @param in_args input arguments
     * @param out output casadi SX expression
     */
    cost(const std::string &name, std::initializer_list<sym> in_args, const cs::SX &out)
        : shared_(new expr_type(name)) {
        assert(out.is_scalar() && "cost output must be a scalar");
        (*this)->set_from_casadi(in_args, out);
    }
    cost() = default;
    using shared_::operator=;
    template <typename derived_impl>
        requires(std::derived_from<derived_impl, impl::cost>)
    /// @brief will get the shared ownership of impl_rval
    cost(derived_impl *impl_rval) : shared_(impl_rval) {}
    /**
     * @brief make state-only cost, appending suffix "_terminal" to costs
     *
     * @tparam derived_impl derived from impl::cost
     * @return the pointer
     */
    template <typename derived_impl = impl::cost>
        requires(std::derived_from<derived_impl, impl::cost>)
    cost &as_terminal() {
        if constexpr (!std::is_same_v<derived_impl, impl::cost>) {
            if (dynamic_cast<derived_impl *>(this->get()) == nullptr) {
                // not the same type, cast
                this->reset(new derived_impl(std::move(**this)));
            }
        }
        *const_cast<std::string *>(&(*this)->name_) += "_terminal";
        (*this)->substitute_x_to_y();
        return *this;
    }
};
}; // namespace moto