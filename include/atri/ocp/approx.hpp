#ifndef __approx__
#define __approx__

// #include <atri/core/sparsity.hpp>
#include <atri/core/fwd.hpp>
#include <atri/ocp/problem.hpp>

namespace atri {

struct raw_data {
    raw_data(problem_ptr_t exprs);

    auto get(sym_ptr_t sym) {
        return sym->make_vec(
            exprs_->get_data_ptr(value_[sym->field_].data(), sym));
    }

    void swap(raw_data &rhs) {
        this->exprs_.swap(rhs.exprs_);
        this->value_.swap(rhs.value_);
    }

    problem_ptr_t exprs_;
    std::array<vector, field::num_sym> value_;
    struct raw_approx {
        vector v_;                                // value
        std::array<matrix, field::num_prim> jac_; // jacobian
    } approx_[field::num_constr];
    // cost
    row_vector jac_[field::num_prim];
    matrix hessian_[field::num_prim][field::num_prim]; // cost hessian
};

/////////////////////////////////////////////////////////////////////

struct approx;

// sparse version
struct sparse_approx_data {
    // use ref to exploit sparsity (avoid copy)
    std::vector<mapped_vector> in_args_;
    vector_ref v_;                // value
    std::vector<matrix_ref> jac_; // jacobian, idx correspond to in_args_
    std::vector<std::vector<matrix_ref>> hess_; // hessian for cost
    // std::vector<sparse_mat> jac_;
    sparse_approx_data(raw_data &raw, std::vector<sym_ptr_t> in_args,
                       approx &f);
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
    virtual void setup_sparsity(sparse_approx_data_ptr_t data) {}

  protected:
    std::vector<sym_ptr_t> in_args_;
    virtual void jacobian_impl(sparse_approx_data_ptr_t data) {
        throw std::runtime_error(
            fmt::format("jacobian not implemented for approx {}", name_));
    };
    virtual void hessian_impl(sparse_approx_data_ptr_t data) {
        throw std::runtime_error(
            fmt::format("hessian not implemented for approx {}", name_));
    };

  public:
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
    virtual sparse_approx_data_ptr_t make_data(raw_data &raw);

    /**
     * @brief evaluate the approx
     * @note currently using template to avoid ifelse, maybe unnecessary
     * @tparam eval_val evaluate value if true
     * @tparam eval_jac evaluate jacobian if true
     * @tparam eval_hess evaluate hessian if true
     */
    void evaluate(problem_ptr_t problem, sparse_approx_data_ptr_t data,
                  bool eval_val, bool eval_jac = false,
                  bool eval_hess = false) {
        // if (eval_jac)
        if (eval_jac)
            jacobian_impl(data);
    }
};
def_ptr(approx);

// example
class foot_kin_constr : public approx {
  private:
    sym_ptr_t q_;
    void jacobian_impl(sparse_approx_data_ptr_t data) override {
        auto q = data->in_args_[0];
    }

  public:
    foot_kin_constr(const std::string &frame, sym_ptr_t q)
        : approx(frame, 3, __eq_cstr_s, approx_order::first), q_(q) {
        in_args_.push_back(q_);
    }
};

} // namespace atri

#endif /*__approx_*/