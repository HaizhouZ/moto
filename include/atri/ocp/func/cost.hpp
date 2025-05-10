// a collection of cost functions;
// class cost : public approx{
// void add_cost
// }
#include <atri/ocp/core/approx.hpp>

namespace atri {
struct cost : public approx {
    cost(const std::string &name, approx_order order = approx_order::second) : approx(name, 1, __cost, order) {}

  protected: // unify interface with constr
    virtual void value(sparse_approx_data_ptr_t) = 0;
    virtual void jacobian(sparse_approx_data_ptr_t) = 0;
    virtual void hessian(sparse_approx_data_ptr_t) = 0;

  private:
    void value_impl(sparse_approx_data_ptr_t d) override final {
        value(d);
    };
    void jacobian_impl(sparse_approx_data_ptr_t d) override final {
        jacobian(d);
    };
    void hessian_impl(sparse_approx_data_ptr_t d) override final {
        hessian(d);
    };
};
}; // namespace atri