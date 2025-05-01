#ifndef __approx__
#define __approx__

#include <atri/ocp/core/problem.hpp>

namespace atri {

struct problem_data; // fwd declaration

/////////////////////////////////////////////////////////////////////

struct approx;

// sparse version
struct sparse_approx_data {
    // use ref to exploit sparsity (avoid copy)
    std::vector<mapped_vector> in_args_;
    vector_ref v_; // value
    // derivatives; index corresponds to in_args_
    /// @note in hess_ only the upper block triangular part are stored!(blocked by field)
    /// @example Q_xu is store instead of Q_ux;
    std::vector<matrix_ref> jac_;               // jacobian, idx correspond to in_args_
    std::vector<std::vector<matrix_ref>> hess_; // hessian for cost
    // std::vector<sparse_mat> jac_;
    sparse_approx_data(problem_data *raw, std::vector<sym_ptr_t> in_args, approx *f);
    sparse_approx_data(const sparse_approx_data &rhs) = delete; // disable this
    sparse_approx_data(sparse_approx_data &&rhs) : v_(rhs.v_) {
        in_args_ = std::move(rhs.in_args_);
        jac_ = std::move(rhs.jac_);
        hess_ = std::move(rhs.hess_);
    }
};

def_ptr(sparse_approx_data);
/////////////////////////////////////////////////////////////////////

class approx : public expr { /// todo: change to differentiable for precompute
  private:
    approx_order order_;
    std::vector<sym_ptr_t> in_args_;
    virtual void setup_sparsity(sparse_approx_data_ptr_t data) {}

  protected:
    virtual void jacobian_impl(sparse_approx_data_ptr_t data) {
        throw std::runtime_error(
            fmt::format("jacobian not implemented for approx {}", name_));
    };
    virtual void hessian_impl(sparse_approx_data_ptr_t data) {
        throw std::runtime_error(
            fmt::format("hessian not implemented for approx {}", name_));
    };

  public:
    void add_argument(sym_ptr_t in) { in_args_.push_back(in); }
    void add_arguments(std::initializer_list<sym_ptr_t> args) {
        in_args_.insert(in_args_.end(), args.begin(), args.end());
    }
    const auto &in_args() { return in_args_; }
    inline approx_order order() { return order_; }

    approx(const std::string &name, size_t dim, field_t field,
           approx_order order)
        : expr(name, dim, field), order_(order) {}

    /**
     * @brief setup the approx_data
     * @details setup primal input ptrs ; allocates value and derivative memory
     * @param raw space shoud have been allocated
     * @return sparse_approx_data_ptr_t
     */
    virtual sparse_approx_data_ptr_t make_data(problem_data *raw);

    /**
     * @brief evaluate the approx
     * @note currently using template to avoid ifelse, maybe unnecessary
     * @tparam eval_val evaluate value if true
     * @tparam eval_jac evaluate jacobian if true
     * @tparam eval_hess evaluate hessian if true
     */
    void evaluate(problem_ptr_t problem, sparse_approx_data_ptr_t data,
                  bool eval_val, bool eval_jac = false, bool eval_hess = false) {
        // if (eval_jac)
        if (eval_jac)
            jacobian_impl(data);
        if (eval_hess)
            hessian_impl(data);
    }
};
def_ptr(approx);
} // namespace atri

#endif /*__approx_*/