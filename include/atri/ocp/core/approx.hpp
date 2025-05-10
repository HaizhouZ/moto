#ifndef __approx__
#define __approx__

#include <atri/ocp/core/problem.hpp>

namespace atri {

struct approx_data; // fwd declaration
struct sym_data;    // fwd declaration

/////////////////////////////////////////////////////////////////////
enum class approx_order { zero = 0,
                          first,
                          second };
struct approx;

/**
 * @brief sparse approximation data
 * the dense data are mapped in the members of this class
 * @note in hess_ only the upper block triangular part are stored!(blocked by field)
 * for example Q_xu is store instead of Q_ux;
 */
struct sparse_approx_data {
    // use ref to exploit sparsity (avoid copy)
    std::vector<mapped_vector> in_args_;
    vector_ref v_; // value
    // jacobian, index correspond to in_args_
    std::vector<matrix_ref> jac_;
    // hessian for cost. index corresponds to in_args_
    std::vector<std::vector<matrix_ref>> hess_;
    /**
     * @brief Construct a new sparse approx data object
     *
     * @param primal sym data including states inputs etc
     * @param raw dense raw data of approximation
     * @param f approximation
     */
    sparse_approx_data(sym_data *primal, approx_data *raw, approx *f);
    sparse_approx_data(const sparse_approx_data &rhs) = delete; // disable this
    sparse_approx_data(sparse_approx_data &&rhs) : v_(rhs.v_), sym_uid_idx_(rhs.sym_uid_idx_) {
        in_args_ = std::move(rhs.in_args_);
        jac_ = std::move(rhs.jac_);
        hess_ = std::move(rhs.hess_);
    }

    auto &operator()(sym in) {
        return in_args_[sym_uid_idx_[in->uid_]];
    }

  private:
    std::unordered_map<size_t, size_t> &sym_uid_idx_;
};

def_ptr(sparse_approx_data);
/////////////////////////////////////////////////////////////////////
/**
 * @brief approximation class for generic functions
 */
class approx : public expr { /// todo: change to differentiable for precompute
  private:
    approx_order order_;
    std::vector<sym> in_args_;
    std::unordered_map<size_t, size_t> sym_uid_idx_;
    friend struct sparse_approx_data;

    /// @todo
    virtual void setup_sparsity(sparse_approx_data &data) {}

  public:
    virtual void value_impl(sparse_approx_data &data) {
        throw std::runtime_error(
            fmt::format("value not implemented for approx {}", name_));
    };
    virtual void jacobian_impl(sparse_approx_data &data) {
        throw std::runtime_error(
            fmt::format("jacobian not implemented for approx {}", name_));
    };
    virtual void hessian_impl(sparse_approx_data &data) {
        throw std::runtime_error(
            fmt::format("hessian not implemented for approx {}", name_));
    };

  public:
    void add_argument(sym in) {
        in_args_.push_back(in);
        sym_uid_idx_[in->uid_] = sym_uid_idx_.size();
    }
    void add_arguments(std::initializer_list<sym> args) {
        for (auto in : args) {
            add_argument(in);
        }
    }
    // get input argument values
    const auto &in_args() { return in_args_; }
    // order of approximation
    inline approx_order order() { return order_; }

    approx(const std::string &name, size_t dim, field_t field,
           approx_order order)
        : expr(name, dim, field), order_(order) {}

    approx(approx &&rhs)
        : expr(std::move(rhs)), order_(rhs.order_),
          in_args_(std::move(rhs.in_args_)),
          sym_uid_idx_(std::move(rhs.sym_uid_idx_)) {}
    /**
     * @brief get other variables related to this approximation
     * @details here it is the input arguments, probably also parameters in the future
     * @return std::vector<expr_ptr_t> list of expressions
     */
    std::vector<expr_ptr_t> get_aux() override {
        return std::vector<expr_ptr_t>(in_args_.begin(), in_args_.end());
    }

    /**
     * @brief setup the sparse approx data
     * @details will setup the mapping from the dense approx_data to sparse_approx_data
     * @return sparse_approx_data_ptr_t
     */
    virtual sparse_approx_data_ptr_t make_data(sym_data *primal, approx_data *raw);

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
        if (eval_val)
            value_impl(*data);
        if (eval_jac)
            jacobian_impl(*data);
        if (eval_hess)
            hessian_impl(*data);
    }
};
def_ptr(approx);
} // namespace atri

#endif /*__approx_*/