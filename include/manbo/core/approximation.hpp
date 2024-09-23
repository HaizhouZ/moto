#ifndef __APPROXIMATION__
#define __APPROXIMATION__

#include <manbo/core/sparsity.hpp>
#include <manbo/core/multivariate.hpp>

namespace manbo {
struct semi_sparse_matrix {
    sparsity_info sp_;

};
struct approx_data {
    multivariate::stacked_const_ptr_t in_ptr_;
    vector v_;  // value
    matrix fv_;
    const sparsity_info& sp_;
    approx_data(size_t n_in, const sparsity_info& info)
        : in_ptr_(n_in, 0), sp_(info) {}
    virtual void AtDA() {} // A^T @ D @ A
    virtual void AtDv() {} // A^T @ D @ v
};

def_ptr(approx_data);

class first_approx : public multivariate {
   protected:
    sparsity_info sp_;

   public:
    first_approx(const std::string& name, size_t dim, field_type field,
                 approx_type level = approx_type::first)
        : multivariate(name, dim, field, level) {}

    auto make_approx_data() {
        return std::make_shared<approx_data>(n_in(), sp_);
    }

    void evaluate(std::shared_ptr<problem> problem,
                  primal_data_ptr_t raw,
                  approx_data_ptr_t data,
                  bool eval_val = true,
                  bool eval_jac = false) {
        get_primal(problem, raw, data->in_ptr_);
        if (eval_jac)
            jacobian_impl(data);
    }
    virtual void jacobian_impl(approx_data_ptr_t data) = 0;
};
// example
class foot_kin_constr : public first_approx {
   private:
    sym_ptr_t q_;

   public:
    foot_kin_constr(const std::string& frame, sym_ptr_t q)
        : first_approx(frame, 3, field_type::constr), q_(q) {
        add_in(q_);
    }
    void jacobian_impl(approx_data_ptr_t data) override {
        auto q = q_->from(data->in_ptr_[0]);
    }
};

struct second_approx : public first_approx {
    second_approx(const std::string& name, size_t dim, field_type field)
        : first_approx(name, dim, field, approx_type::second) {}
    virtual void compute_hessian(Eigen::Ref<vector> x) {
    }
};
}  // namespace manbo

#endif /*__APPROXIMATION_*/