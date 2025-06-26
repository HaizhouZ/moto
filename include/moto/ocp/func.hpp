#ifndef __approx__
#define __approx__

#include <moto/ocp/approx_storage.hpp>
#include <moto/ocp/problem.hpp>
#include <moto/ocp/sym_data.hpp>

namespace moto {
class shared_data; // forward declaration

enum class approx_order { zero = 0,
                          first,
                          second,
                          none };
struct func_impl;
def_ptr(func_impl);
/////////////////////////////////////////////////////////////////////
/**
 * @brief sparse primal data
 * sparse view to access the input arguments of a function
 * @note it uses reference to avoid copying the input arguments
 */
struct sp_arg_map {
    /**
     * @brief Construct a new sparse primal data object
     * @param primal sym data including states inputs etc
     * @param shared shared data
     * @param f function implementation pointer
     */
    sp_arg_map(sym_data &primal, shared_data &shared, func_impl &f);
    // constructor for sparse primal data with vector_ref
    sp_arg_map(std::vector<vector_ref> &&primal, shared_data &shared, func_impl &f);

    virtual ~sp_arg_map() = default;
    func_impl &f_;        ///< pointer to the func
    shared_data &shared_; ///< ref to shared data
    /**
     * @brief get the input argument values
     * @note this is a wrapper of in_args_ to access the values
     * @return vector_ref of input arguments
     */
    auto &operator()(const sym &in) {
        return in_args_[sym_uid_idx_[in->uid_]];
    }

    const auto &in_args() const { return in_args_; }
    auto in_args(size_t i) const { return in_args_[i]; }

  protected:
    /// use ref to exploit sparsity (avoid copy)
    std::vector<vector_ref> in_args_;
    std::unordered_map<size_t, size_t> &sym_uid_idx_;
};
def_unique_ptr(sp_arg_map);
/////////////////////////////////////////////////////////////////////
/**
 * @brief sparse approximation data
 * the dense data are mapped in the members of this class
 * @note in hess_ only the upper block triangular part are stored!(blocked by field)
 * for example Q_ux is store instead of Q_ux;
 */
struct sp_approx_map : public sp_arg_map {

    vector_ref v_; ///< value ref
    /// jacobian, by default index correspond to @ref sp_arg_map::in_args_
    std::vector<matrix_ref> jac_;
    /// hessian for cost. index corresponds to @ref sp_arg_map::in_args_
    std::vector<std::vector<matrix_ref>> hess_;
    /**
     * @brief Construct a new sparse approx data object
     *
     * @param primal sym data including states inputs etc
     * @param raw dense raw data of approximation
     * @param shared shared data
     * @param f approximation
     */
    sp_approx_map(sym_data &primal, approx_storage &raw, shared_data &shared, func_impl &f);
    /**
     * @brief Construct a new sparse approx data object
     *
     * @param primal sym data including states inputs etc
     * @param v value vector reference
     * @param jac jacobian matrix references
     * @param shared shared data
     * @param f approximation function implementation pointer (unique_ptr const ref)
     * @note this constructor is used for approximations not mapped from @ref approx_storage
     */
    sp_approx_map(sym_data &primal, vector_ref v, const std::vector<matrix_ref> &jac, shared_data &shared, func_impl &f);
    /// constructor for sparse approx data with already created sp_arg_map
    sp_approx_map(sp_arg_map &&rhs, vector_ref v, std::vector<matrix_ref> &&jac)
        : sp_arg_map(std::move(rhs)), v_(v), jac_(std::move(jac)) {}
};

def_unique_ptr(sp_approx_map);
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

template <typename... data_type>
auto make_composed(data_type &&...other_data) {
    return std::make_unique<composed_data<data_type...>>(std::forward<data_type>(other_data)...);
}
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
    friend struct sp_arg_map;

    /// @todo to be implemented
    virtual void setup_sparsity([[maybe_unused]] sp_approx_map &data) {}

  public:
    virtual void value_impl([[maybe_unused]] sp_approx_map &data) { value(data); };
    virtual void jacobian_impl([[maybe_unused]] sp_approx_map &data) { jacobian(data); };
    virtual void hessian_impl([[maybe_unused]] sp_approx_map &data) { hessian(data); };

  public:
    void add_argument(const sym &in) {
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
    /// @brief get input argument values
    const auto &in_args() const { return in_args_; }
    /// @brief order of approximation
    inline approx_order order() { return order_; }
    /**
     * @breid check if the function is an approximation
     * @note funcs in __undefined will be considered as (possibly) approximation
     * @return true if field is not __pre_comp or __usr_func and field value is greater than __dyn
     */
    bool is_approx() const { return (field_ != __pre_comp && field_ != __usr_func) && field_ > __dyn; }

    /**
     * @brief get other variables related to this approximation
     * @details here it is the input arguments, probably also parameters in the future
     * @return list of expressions
     */
    expr_list *get_aux() override { return &in_args_; }
    /**
     * @brief setup the sparse func data
     * @details will setup the mapping from the dense approx_storage to sp_approx_map
     * @return sp_approx_map_ptr_t
     */
    virtual sp_approx_map_ptr_t make_approx_data_mapping(sym_data &primal,
                                                         approx_storage &raw,
                                                         shared_data &shared);
    /**
     * @brief evaluate the func
     * @param data sp_approx_map to be evaluated
     * @param eval_val evaluate value if true
     * @param eval_jac evaluate jacobian if true
     * @param eval_hess evaluate hessian if true
     */
    void evaluate_approx(sp_approx_map &data,
                         bool eval_val, bool eval_jac = false, bool eval_hess = false) {
        if (eval_val)
            value_impl(data);
        if (eval_jac)
            jacobian_impl(data);
        if (eval_hess)
            hessian_impl(data);
    }
    template <typename T>
    void evaluate_approx(T &data,
                         bool eval_val, bool eval_jac = false, bool eval_hess = false) {
        evaluate_approx(dynamic_cast<sp_approx_map &>(data), eval_val, eval_jac, eval_hess);
    }
    /**
     * @brief load the external approximation functions
     * @note will find the functions with suffix _jac, _hess
     * @param path folder of the external functions, default is "gen"
     */
    virtual void load_external(const std::string &path = "gen");

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
        // make a default make_data
        make_data = [this](sym_data &primal, shared_data &shared) {
            return std::make_unique<sp_arg_map>(primal, shared, *this);
        };
    }
    /**
     * @brief constructor for approximation functions
     * @note the order must be specified, otherwise it will be considered as non-approximation
     * @param name name of the function
     * @param order order of the approximation
     * @param dim dimension of the function, default is 0
     * @param field field type, default is __undefined (to be finalized later, @ref finalize_impl)
     */
    func_impl(const std::string &name, approx_order order, size_t dim = 0, field_t field = __undefined)
        : expr_impl(name, dim, field), order_(order) {
    }

  public:
    /// @brief callback to evaluate the value of the approximation
    std::function<void(sp_approx_map &)> value;
    /// @brief callback to evaluate the jacobian of the approximation
    std::function<void(sp_approx_map &)> jacobian;
    /// @brief callback to evaluate the hessian of the approximation
    std::function<void(sp_approx_map &)> hessian;
    /// @brief callback to call a non-approximation function
    std::function<void(sp_arg_map &)> call;
    /// @brief callback to make data（for non-approx) @note will not be called in @ref make_approx_data_mapping
    std::function<sp_arg_map_ptr_t(sym_data &, shared_data &)> make_data;
};
/////////////////////////////////////////////////////////////////////
/**
 * @brief shared data for all funcs in one problem formulation
 * @note it will store (class derived from) sp_arg_map
 */
class shared_data {
    std::unordered_map<size_t, sp_arg_map_ptr_t> data_;

  public:
    shared_data(const ocp_ptr_t &prob, sym_data &primal);

    ocp_ptr_t prob_;
    /// @brief add data by uid of the func (owner of the data)
    void add(size_t uid, sp_arg_map_ptr_t &&data) { data_.try_emplace(uid, std::move(data)); }
    /// @brief add data by func shared pointer (owner of the data)
    template <typename derived>
        requires std::is_base_of_v<func_impl, derived>
    void add(const std::shared_ptr<derived> &expr, sp_arg_map_ptr_t &&data) {
        assert(expr->field_ == __pre_comp || expr->field_ == __usr_func);
        add(expr->uid, std::move(data));
    }
    /// @brief get the data by uid
    auto &get(size_t uid) { return *data_.at(uid); }
    /// @brief get the data by func shared pointer (owner of the data)
    template <typename derived>
        requires std::is_base_of_v<func_impl, derived>
    auto &operator()(const std::shared_ptr<derived> &expr) { return get(expr->uid_); }
    template <typename derived>
        requires std::is_base_of_v<func_impl, derived>
    auto &operator()(const derived &expr) { return get(expr.uid_); }
};
def_unique_ptr(shared_data);
} // namespace moto

#endif /*__approx_*/