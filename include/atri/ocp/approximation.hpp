#ifndef __APPROXIMATION__
#define __APPROXIMATION__

#include <atri/core/sparsity.hpp>
#include <atri/ocp/expr_collection.hpp>

namespace atri {
typedef std::array<scalar_t*, field::num_sym> primal_data_ptr_t;

struct approx_data {
    std::vector<const scalar_t*> in_ptr_;
    vector v_;  // value
    std::vector<sparse_mat> jac_;
    approx_data(size_t n_in)
        : in_ptr_(n_in, 0) {}
    approx_data(const approx_data& rhs) = delete;  // disable this

    virtual void AtDA() {}  // A^T @ D @ A
    virtual void AtDv() {}  // A^T @ D @ v
};

def_ptr(approx_data);
/////////////////////////////////////////////////////////////////////

class approximation : public expr {  /// todo: change to differentiable for precompute
   private:
    approx_type level_;

   public:
    inline approx_type approx_level() { return level_; }

    approximation(const std::string& name, size_t dim, field_type field, approx_type level)
        : expr(name, dim, field), level_(level) {}

    virtual approx_data_ptr_t make_approx_data() = 0;

    template <bool eval_val, bool eval_jac = false, bool eval_hess = false>
    void evaluate(std::shared_ptr<expr_collection> expr_collection,
                  primal_data_ptr_t raw,
                  approx_data_ptr_t data) {
        // if (eval_jac)
        if constexpr (eval_jac)
            jacobian_impl(data);
    }

    virtual void jacobian_impl(approx_data_ptr_t data) {
        throw std::runtime_error(fmt::format(
            "jacobian not implemented for approximation {}", name_));
    };
    virtual void hessian_impl(approx_data_ptr_t data) {
        throw std::runtime_error(fmt::format(
            "hessian not implemented for approximation {}", name_));
    };
    inline auto get_primal(expr_collection_ptr_t expr_collection, primal_data_ptr_t raw, sym_ptr_t sym) {
        return sym->from(expr_collection->get_data_ptr(raw[sym->field_], sym));
    }
};
def_ptr(approximation);

// example
class foot_kin_constr : public approximation {
   private:
    sym_ptr_t q_;

   public:
    approx_data_ptr_t make_approx_data() { return nullptr; }
    foot_kin_constr(const std::string& frame, sym_ptr_t q)
        : approximation(frame, 3, field_type::constr, approx_type::first), q_(q) {
    }

    void jacobian_impl(approx_data_ptr_t data) override {
        auto q = q_->from(data->in_ptr_[0]);
    }
};

}  // namespace atri

#endif /*__APPROXIMATION_*/