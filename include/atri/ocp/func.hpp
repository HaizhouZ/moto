#ifndef __approx__
#define __approx__

#include <atri/ocp/approx_storage.hpp>
#include <atri/ocp/problem.hpp>
#include <atri/ocp/sym_data.hpp>

namespace atri {
class shared_data; // forward declaration

enum class approx_order { zero = 0,
                          first,
                          second,
                          none };
struct func_impl;
def_ptr(func_impl);
/////////////////////////////////////////////////////////////////////
struct sparse_primal_data {
    // use ref to exploit sparsity (avoid copy)
    std::vector<vector_ref> in_args_;
    sparse_primal_data(sym_data *primal, shared_data *shared, const func_impl_ptr_t &f)
        : sparse_primal_data(primal, shared, f.get()) {}
    sparse_primal_data(sym_data *primal, shared_data *shared, func_impl *f);
    sparse_primal_data(const sparse_primal_data &rhs) = delete; // disable this
    sparse_primal_data(sparse_primal_data &&rhs)
        : in_args_(std::move(rhs.in_args_)),
          sym_uid_idx_(rhs.sym_uid_idx_),
          f_(rhs.f_),
          shared_(rhs.shared_) {
    }
    virtual ~sparse_primal_data() = default;
    const func_impl *f_;  ///< pointer to the func
    shared_data *shared_; ///< pointer to shared data

    auto &operator()(const sym &in) {
        return in_args_[sym_uid_idx_[in->uid_]];
    }

  protected:
    std::unordered_map<size_t, size_t> &sym_uid_idx_;
};
def_unique_ptr(sparse_primal_data);
/////////////////////////////////////////////////////////////////////
/**
 * @brief sparse approximation data
 * the dense data are mapped in the members of this class
 * @note in hess_ only the upper block triangular part are stored!(blocked by field)
 * for example Q_ux is store instead of Q_ux;
 */
struct sparse_approx_data : public sparse_primal_data {

    vector_ref v_; // value
    // jacobian, index correspond to in_args_
    std::vector<matrix_ref> jac_;
    // hessian for cost. index corresponds to in_args_
    std::vector<std::vector<matrix_ref>> hess_;
    /**
     * @brief Construct a new sparse func data object
     *
     * @param primal sym data including states inputs etc
     * @param raw dense raw data of approximation
     * @param f approximation
     */
    sparse_approx_data(sym_data *primal, approx_storage *raw, shared_data *shared, func_impl *f);
    sparse_approx_data(sym_data *primal, vector_ref v, const std::vector<matrix_ref> &jac, shared_data *shared, const func_impl_ptr_t &f)
        : sparse_approx_data(primal, v, jac, shared, f.get()) {}
    sparse_approx_data(sym_data *primal, vector_ref v, const std::vector<matrix_ref> &jac, shared_data *shared, func_impl *f);
    sparse_approx_data(const sparse_approx_data &rhs) = delete; // disable this
    sparse_approx_data(sparse_approx_data &&rhs) : v_(rhs.v_), sparse_primal_data(std::move(rhs)) {
        jac_ = std::move(rhs.jac_);
        hess_ = std::move(rhs.hess_);
    }
};

def_unique_ptr(sparse_approx_data);
/////////////////////////////////////////////////////////////////////
/**
 * @brief composing several data types into one type
 * it will automatically call to the moving constructor of the data types
 * @tparam data_type must be move constructible
 */
template <typename... data_type>
struct composed_data : public data_type... {
    composed_data(data_type &&...other_data)
        : data_type(std::forward<data_type>(other_data))... {
        static_assert((std::is_move_constructible<data_type>::value && ...),
                      "All data types must be move constructible");
    }
};
/**
 * @brief compose (append) several data types to sparse_primal_data by moving
 * @ref composed_data
 * @tparam data_type must be move constructible
 */
template <typename... data_type>
using composed_primal_data = composed_data<sparse_primal_data, data_type...>;
/**
 * @brief compose (append) several data types to sparse_approx_data by moving
 * @ref composed_data
 * @tparam data_type must be move constructible
 */
template <typename... data_type>
using composed_approx_data = composed_data<sparse_approx_data, data_type...>;
/////////////////////////////////////////////////////////////////////
/**
 * @brief approximation class for generic functions
 * @todo: change to differentiable for precompute
 */
class func_impl : public expr_impl {
  protected:
    approx_order order_;
    expr_list in_args_;
    std::unordered_map<size_t, size_t> sym_uid_idx_;
    friend struct sparse_primal_data;

    /// @todo to be implemented
    virtual void setup_sparsity([[maybe_unused]] sparse_approx_data &data) {}

  public:
    virtual void value_impl([[maybe_unused]] sparse_approx_data &data) { value(data); };
    virtual void jacobian_impl([[maybe_unused]] sparse_approx_data &data) { jacobian(data); };
    virtual void hessian_impl([[maybe_unused]] sparse_approx_data &data) { hessian(data); };

  public:
    void add_argument(const sym& in) {
        in_args_.push_back(in);
        sym_uid_idx_[in->uid_] = sym_uid_idx_.size();
    }
    template <typename T>
    void add_arguments(const T &args) {
        for (const auto &in : args) {
            add_argument(in);
        }
    }

    void add_arguments(std::initializer_list<sym> args) {
        for (const auto &in : args) {
            add_argument(in);
        }
    }
    // get input argument values
    const auto &in_args() const { return in_args_; }
    // order of approximation
    inline approx_order order() { return order_; }
    // check if the function is an approximation, true if yes
    bool is_approx() const {
        return (field_ != __pre_comp && field_ != __usr_func) && field_ > __dyn;
    }
    /**
     * @brief constructor for non-approximation functions
     *
     * @param name name of the function
     * @param field field, must explicitly belong to the non-approximation fields
     */
    func_impl(const std::string &name, field_t field = __undefined)
        : func_impl(name, approx_order::none, 0, field) {
        if (is_approx()) {
            throw std::runtime_error(fmt::format("func {} field type {} not qualified as non-approx",
                                                 name_, magic_enum::enum_name(field)));
        }
    }

    func_impl(const std::string &name, approx_order order, size_t dim = 0, field_t field = __undefined)
        : expr_impl(name, dim, field), order_(order) {
    }

    /**
     * @brief get other variables related to this approximation
     * @details here it is the input arguments, probably also parameters in the future
     * @return list of expressions
     */
    expr_list *get_aux() override {
        return &in_args_;
    }
    /**
     * @brief setup the sparse func data
     * @details will setup the mapping from the dense approx_storage to sparse_approx_data
     * @return sparse_approx_data_ptr_t
     */
    virtual sparse_approx_data_ptr_t make_approx_data_mapping(sym_data *primal,
                                                              approx_storage *raw,
                                                              shared_data *shared);
    /**
     * @brief evaluate the func
     * @note currently using template to avoid ifelse, maybe unnecessary
     * @tparam eval_val evaluate value if true
     * @tparam eval_jac evaluate jacobian if true
     * @tparam eval_hess evaluate hessian if true
     */
    void evaluate_approx(sparse_primal_data &data,
                         bool eval_val, bool eval_jac = false, bool eval_hess = false) {
        assert(dynamic_cast<sparse_approx_data *>(&data) != nullptr); // check only in debug mode
        if (eval_val)
            value_impl(static_cast<sparse_approx_data &>(data));
        if (eval_jac)
            jacobian_impl(static_cast<sparse_approx_data &>(data));
        if (eval_hess)
            hessian_impl(static_cast<sparse_approx_data &>(data));
    }
    /**
     * @brief load the external approximation functions
     * @note will find the functions with suffix _jac, _hess
     * @param path folder of the external functions, default is "gen"
     */
    virtual void load_external(const std::string &path = "gen");

  public:
    std::function<void(sparse_approx_data &)> value;
    std::function<void(sparse_approx_data &)> jacobian;
    std::function<void(sparse_approx_data &)> hessian;
    std::function<void(sparse_primal_data &)> call;
    std::function<sparse_primal_data_ptr_t(sym_data *, shared_data *)> make_data;
};
/////////////////////////////////////////////////////////////////////
struct pre_compute_impl : public func_impl {
    pre_compute_impl(const std::string &name)
        : func_impl(name, __pre_comp) {
    }
};
def_ptr(pre_compute_impl);
/////////////////////////////////////////////////////////////////////
struct pre_compute : public pre_compute_impl_ptr_t {
    pre_compute(const std::string &name)
        : pre_compute_impl_ptr_t(new pre_compute_impl(name)) {
    }
    pre_compute(pre_compute_impl *impl)
        : pre_compute_impl_ptr_t(impl) {
    }
};
/////////////////////////////////////////////////////////////////////
struct usr_func_impl : public func_impl {
    usr_func_impl(const std::string &name, approx_order order, size_t dim = 0)
        : func_impl(name, order, dim, __usr_func) {
    }
};
def_ptr(usr_func_impl);
/////////////////////////////////////////////////////////////////////
struct usr_func : public usr_func_impl_ptr_t {
    usr_func(const std::string &name, approx_order order, size_t dim = 0)
        : usr_func_impl_ptr_t(new usr_func_impl(name, order, dim)) {
    }
    usr_func(usr_func_impl *impl)
        : usr_func_impl_ptr_t(impl) {
    }
};
/////////////////////////////////////////////////////////////////////
class shared_data {
    std::unordered_map<size_t, sparse_primal_data_ptr_t> data_;
    friend struct node_data; // allow node_data to access private members

  public:
    shared_data(const problem_ptr_t &prob, sym_data *primal);

    problem_ptr_t prob_;

    void add(size_t uid, sparse_primal_data_ptr_t &&data) {
        data_.try_emplace(uid, std::move(data));
    }

    void add(expr_impl *f, sparse_primal_data_ptr_t &&data) {
        add(f->uid_, std::move(data));
    }

    template <typename derived>
        requires std::is_base_of_v<func_impl, derived>
    void add(const std::shared_ptr<derived> &expr, sparse_primal_data_ptr_t &&data) {
        add(expr.get(), std::move(data));
    }

    auto &get(size_t uid) {
        return *data_.at(uid);
    }
    auto &get(expr_impl *f) {
        assert(f->field_ == __pre_comp || f->field_ == __usr_func);
        return get(f->uid_);
    }
    template <typename derived>
        requires std::is_base_of_v<func_impl, derived>
    auto &get(const std::shared_ptr<derived> &expr) {
        return get(expr.get());
    }
    auto &get(const std::string &name) {
        return get(expr_index::get(name).get());
    }
};
def_unique_ptr(shared_data);
} // namespace atri

#endif /*__approx_*/