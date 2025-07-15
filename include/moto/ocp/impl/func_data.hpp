#ifndef MOTO_OCP_IMPL_FUNC_DATA_HPP
#define MOTO_OCP_IMPL_FUNC_DATA_HPP

#include <moto/ocp/impl/approx_storage.hpp>
#include <moto/ocp/impl/sym_data.hpp>

namespace moto {
struct sp_arg_map;
def_unique_ptr(sp_arg_map);
namespace impl {
class func;
}
/**
 * @brief shared data for all funcs in one problem formulation
 * @note it will store (class derived from) sp_arg_map
 */
class shared_data {
    std::unordered_map<size_t, sp_arg_map_ptr_t> data_; ///< [uid (of data owner), data]

  public:
    shared_data(const ocp_ptr_t &prob, sym_data &primal);

    ocp_ptr_t prob_;
    /// @brief add data by uid of the func (owner of the data)
    void add(size_t uid, sp_arg_map_ptr_t &&data) { data_.try_emplace(uid, std::move(data)); }
    /// @brief add data by func shared pointer (owner of the data)
    template <typename derived>
        requires std::is_base_of_v<impl::func, derived>
    void add(const std::shared_ptr<derived> &expr, sp_arg_map_ptr_t &&data) {
        assert(expr->field_ == __pre_comp || expr->field_ == __usr_func);
        add(expr->uid, std::move(data));
    }
    /// @brief get the data by uid
    auto &get(size_t uid) { return *data_.at(uid); }
    /// @brief get the data by func shared pointer (owner of the data) by uid
    template <typename derived>
        requires std::is_base_of_v<impl::func, derived>
    auto &operator[](const std::shared_ptr<derived> &expr) { return get(expr->uid_); }
    /// @brief get the data of the func (by uid)
    template <typename derived>
        requires std::is_base_of_v<impl::func, derived>
    auto &operator[](const derived &expr) { return get(expr.uid_); }
};
def_unique_ptr(shared_data);
/////////////////////////////////////////////////////////////////////

/// @brief approximation order enum
/// @details used to indicate the order of approximation, e.g., first order, second order
/// @note `none` is used for non-approximation functions, e.g., pre-compute functions
enum class approx_order { none = 0,
                          zero,
                          first,
                          second };
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
    sp_arg_map(sym_data &primal, shared_data &shared, impl::func &f);
    // constructor for sparse primal data with vector_ref
    sp_arg_map(std::vector<vector_ref> &&primal, shared_data &shared, impl::func &f);

    virtual ~sp_arg_map() = default;
    impl::func &func_;    ///< pointer to the func
    shared_data &shared_; ///< ref to shared data
    /**
     * @brief get the input argument values
     * @note this is a wrapper of in_args_ to access the values
     * @return vector_ref of input arguments
     */
    auto operator[](const sym &in) {
        return in_args_[sym_uid_idx_.at(in->uid_)];
    }
    /// @brief get the input argument values by index
    auto operator[](size_t i) const { return in_args_.at(i); }

    const auto &in_arg_data() const { return in_args_; }

    const auto &problem() const { return shared_.prob_; }

  protected:
    /// use ref to exploit sparsity (avoid copy)
    std::vector<vector_ref> in_args_;
    std::unordered_map<size_t, size_t> &sym_uid_idx_;
};
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
    sp_approx_map(sym_data &primal, approx_storage &raw, shared_data &shared, impl::func &f);
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
    sp_approx_map(sym_data &primal, vector_ref v, const std::vector<matrix_ref> &jac, shared_data &shared, impl::func &f);
    /// constructor for sparse approx data with already created sp_arg_map
    sp_approx_map(sp_arg_map &&rhs, vector_ref v, std::vector<matrix_ref> &&jac)
        : sp_arg_map(std::move(rhs)), v_(v), jac_(std::move(jac)) {}

    auto jac(const sym &in) const {
        return jac_[sym_uid_idx_.at(in->uid_)];
    }
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
} // namespace moto

#endif // MOTO_OCP_IMPL_FUNC_DATA_HPP