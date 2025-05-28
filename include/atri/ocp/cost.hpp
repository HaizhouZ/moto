// a expr_list of cost_impl functions;
// class cost_impl : public approx{
// void add_cost_impl
// }
#include <atri/ocp/approx.hpp>

namespace atri {
struct cost_impl : public approx {
    cost_impl(const std::string &name, approx_order order = approx_order::second)
        : approx(name, order, 1, __cost) {
        value = [this](auto &d) { approx::value_impl(d); };
        jacobian = [this](auto &d) { approx::jacobian_impl(d); };
        hessian = [this](auto &d) { approx::hessian_impl(d); };
    }
    cost_impl(cost_impl &&rhs)
        : approx(std::move(rhs)),
          value(std::move(rhs.value)),
          jacobian(std::move(rhs.jacobian)),
          hessian(std::move(rhs.hessian)) {}

    void load_external(const std::string &path = "gen");

  protected: // unify interface with constr
    std::function<void(sparse_approx_data &)> value;
    std::function<void(sparse_approx_data &)> jacobian;
    std::function<void(sparse_approx_data &)> hessian;

  private:
    void value_impl(sparse_approx_data &d) override final {
        value(d);
    };
    void jacobian_impl(sparse_approx_data &d) override final {
        jacobian(d);
    };
    void hessian_impl(sparse_approx_data &d) override final {
        hessian(d);
    };
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
    return cst;
}
/**
 * @brief wrapper of cost_impl, in fact a pointer
 *
 */
struct cost : public cost_impl_ptr_t {
    cost(const std::string &name)
        : cost_impl_ptr_t(new cost_impl(name)) {
    }
    cost(cost_impl &&impl) : cost_impl_ptr_t(new cost_impl(std::move(impl))) {}
    cost(const cost &rhs) = default;
};
}; // namespace atri