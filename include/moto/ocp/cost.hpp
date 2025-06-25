#include <moto/ocp/func.hpp>

namespace moto {
struct cost_impl : public func_impl {
    cost_impl(const std::string &name, approx_order order = approx_order::second)
        : func_impl(name, order, 1, __cost) {
        value = [this](auto &d) { func_impl::value_impl(d); };
        jacobian = [this](auto &d) { func_impl::jacobian_impl(d); };
        hessian = [this](auto &d) { func_impl::hessian_impl(d); };
    }
};
def_ptr(cost_impl);
/**
 * @brief helper function, appending suffix "_terminal" to costs
 *
 * @tparam cost_type derived from cost_impl
 * @param cst pointer from new cost_type(...)
 * @return the pointer
 */
template <typename cost_type>
    requires std::is_base_of_v<cost_impl, cost_type>
inline auto make_terminal_cost(cost_type *cst) {
    *const_cast<std::string *>(&cst->name_) = cst->name_ + "_terminal";
    return std::shared_ptr<cost_type>(cst);
}
/**
 * @brief wrapper of cost_impl, in fact a pointer
 *
 */
struct cost : public cost_impl_ptr_t {
    cost(const std::string &name)
        : cost_impl_ptr_t(new cost_impl(name)) {
    }
    cost(cost_impl *impl) : cost_impl_ptr_t(impl) {}
    cost(const cost &rhs) = default;
};
}; // namespace moto