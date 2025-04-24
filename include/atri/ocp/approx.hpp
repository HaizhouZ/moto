#ifndef __approx__
#define __approx__

#include <atri/core/sparsity.hpp>
#include <atri/ocp/expr_sets.hpp>

namespace atri {

class primal_data {
  private:
    expr_sets_ptr_t exprs_;

  public:
    std::array<std::vector<vector>, field::num_sym> value_;
    auto &get(sym_ptr_t sym) {
        return value_[sym->field_][exprs_->idx_[sym->uid_]];
    }
    primal_data(expr_sets_ptr_t exprs) : exprs_(exprs) {
        for (size_t i = 0; i < field::num_sym; i++) {
            auto &v = value_[i];
            for (auto &e : exprs_->expr_[i]) {
                v.push_back(vector(e->dim_));
            }
        }
    }
    void swap(primal_data &rhs) {
        this->exprs_.swap(rhs.exprs_);
        this->value_.swap(rhs.value_);
    }
};

/////////////////////////////////////////////////////////////////////

struct approx_data {
    std::vector<vector_ref> in_args_;
    vector v_;                 // value
    std::vector<matrix> jac_;  // jacobian
    std::vector<matrix> hess_; // hessian
    // std::vector<sparse_mat> jac_;
    approx_data(primal_data raw, std::vector<sym_ptr_t> in_args, size_t dim,
                bool jac = false, bool hess = false) {
        for (size_t i = 0; i < in_args.size(); i++) {
            auto arg = in_args[i];
            in_args_.push_back(raw.get(arg));
        }
        v_.resize(dim);
        if (jac) {
            jac_.resize(in_args_.size());
            for (size_t i = 0; i < in_args_.size(); i++) {
                jac_[i].resize(dim, in_args[i]->dim_);
            }
        }
        if (hess) {
            if (dim == 1) {
                hess_.resize(in_args_.size());
                for (size_t i = 0; i < in_args_.size(); i++) {
                    hess_[i].resize(in_args[i]->dim_, in_args[i]->dim_);
                    // todo : cross hessian
                }
            }
        }
    }
    approx_data(const approx_data &rhs) = delete; // disable this
};

def_ptr(approx_data);
/////////////////////////////////////////////////////////////////////

class approx : public expr { /// todo: change to differentiable for precompute
  private:
    approx_order order_;
    virtual void setup_sparsity(approx_data_ptr_t data) {}

  protected:
    std::vector<sym_ptr_t> in_args_;
    virtual void jacobian_impl(approx_data_ptr_t data) {
        throw std::runtime_error(
            fmt::format("jacobian not implemented for approx {}", name_));
    };
    virtual void hessian_impl(approx_data_ptr_t data) {
        throw std::runtime_error(
            fmt::format("hessian not implemented for approx {}", name_));
    };

  public:
    inline approx_order order() { return order_; }

    approx(const std::string &name, size_t dim, field::type field,
           approx_order order)
        : expr(name, dim, field), order_(order) {}

    /**
     * @brief setup the approx_data
     * @details setup primal input ptrs ; allocates value and derivative memory
     * @param raw space shoud have been allocated
     * @return approx_data_ptr_t
     */
    approx_data_ptr_t make_data(primal_data &raw) {
        auto data = approx_data_ptr_t();
        data.reset(new approx_data(raw, in_args_, dim_,
                                   order_ >= approx_order::first,
                                   order_ >= approx_order::second));
        for (size_t i = 0; i < in_args_.size(); i++) {
            auto arg = in_args_[i];
        }
        setup_sparsity(data);
        return data;
    }

    /**
     * @brief evaluate the approx
     * @note currently using template to avoid ifelse, maybe unnecessary
     * @tparam eval_val evaluate value if true
     * @tparam eval_jac evaluate jacobian if true
     * @tparam eval_hess evaluate hessian if true
     */
    template <bool eval_val, bool eval_jac = false, bool eval_hess = false>
    void evaluate(expr_sets_ptr_t expr_sets, approx_data_ptr_t data) {
        // if (eval_jac)
        if constexpr (eval_jac)
            jacobian_impl(data);
    }
};
def_ptr(approx);

// example
class foot_kin_constr : public approx {
  private:
    sym_ptr_t q_;
    void jacobian_impl(approx_data_ptr_t data) override {
        auto q = data->in_args_[0];
    }

  public:
    foot_kin_constr(const std::string &frame, sym_ptr_t q)
        : approx(frame, 3, field::type::constr, approx_order::first), q_(q) {
        in_args_.push_back(q_);
    }
};

} // namespace atri

#endif /*__approx_*/