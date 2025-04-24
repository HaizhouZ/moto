#ifndef __approx__
#define __approx__

#include <atri/core/sparsity.hpp>
#include <atri/ocp/expr_set.hpp>

namespace atri {
typedef std::array<scalar_t *, field::num_sym> primal_data_ptr_t;

struct approx_data {
    std::vector<const scalar_t *> in_ptr_;
    vector v_; // value
    std::vector<sparse_mat> jac_;
    approx_data(size_t n_in) : in_ptr_(n_in, 0) {}
    approx_data(const approx_data &rhs) = delete; // disable this

    virtual void AtDA() {} // A^T @ D @ A
    virtual void AtDv() {} // A^T @ D @ v
};

def_ptr(approx_data);
/////////////////////////////////////////////////////////////////////

class approx : public expr { /// todo: change to differentiable for precompute
  private:
    approx_order order_;

  public:
    inline approx_order order() { return order_; }

    approx(const std::string &name, size_t dim, field::type field,
           approx_order order)
        : expr(name, dim, field), order_(order) {}

    virtual approx_data_ptr_t make_data() = 0;

    template <bool eval_val, bool eval_jac = false, bool eval_hess = false>
    void evaluate(expr_sets_ptr_t expr_sets, primal_data_ptr_t raw,
                  approx_data_ptr_t data) {
        // if (eval_jac)
        if constexpr (eval_jac)
            jacobian_impl(data);
    }

    virtual void jacobian_impl(approx_data_ptr_t data) {
        throw std::runtime_error(
            fmt::format("jacobian not implemented for approx {}", name_));
    };
    virtual void hessian_impl(approx_data_ptr_t data) {
        throw std::runtime_error(
            fmt::format("hessian not implemented for approx {}", name_));
    };
    inline auto get_primal(expr_sets_ptr_t exprs, primal_data_ptr_t raw,
                           sym_ptr_t sym) {
        return sym->make_vec(exprs->get_data_ptr(raw[sym->field_], sym));
    }
};
def_ptr(approx);

// example
class foot_kin_constr : public approx {
  private:
    sym_ptr_t q_;

  public:
    approx_data_ptr_t make_data() { return nullptr; }
    foot_kin_constr(const std::string &frame, sym_ptr_t q)
        : approx(frame, 3, field::type::constr, approx_order::first), q_(q) {}

    void jacobian_impl(approx_data_ptr_t data) override {
        auto q = q_->make_vec(data->in_ptr_[0]);
    }
};

} // namespace atri

#endif /*__approx_*/