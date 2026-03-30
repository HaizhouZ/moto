#ifndef MOTO_OCP_IMPL_FUNC_DATA_HPP
#define MOTO_OCP_IMPL_FUNC_DATA_HPP

#include <moto/ocp/impl/merit_data.hpp>
#include <moto/ocp/impl/sym_data.hpp>

namespace moto {
struct func_arg_map;
def_unique_ptr(func_arg_map);
class generic_func;
/**
 * @brief shared data for all funcs in one problem formulation
 * @note it will store (class derived from) func_arg_map
 */
class shared_data {
    std::unordered_map<size_t, func_arg_map_ptr_t> data_; ///< [uid (of data owner), data]

  public:
    shared_data(ocp *prob, sym_data *primal, merit_data *raw = nullptr);
    shared_data(shared_data &&) = default;

    const ocp *prob_;
    /// @brief add data by uid of the func (owner of the data)
    void add(size_t uid, func_arg_map_ptr_t &&data) { data_.try_emplace(uid, std::move(data)); }
    /// @brief add data by func shared pointer (owner of the data)
    template <typename derived>
        requires std::is_base_of_v<generic_func, derived>
    void add(const derived &ex, func_arg_map_ptr_t &&data) {
        assert(ex.field() == __pre_comp || ex.field() == __usr_func);
        add(ex.uid(), std::move(data));
    }
    auto *try_get(size_t uid) {
        auto it = data_.find(uid);
        if (it == data_.end())
            return static_cast<func_arg_map *>(nullptr);
        return it->second.get();
    }
    /// @brief get the data by uid
    auto &get(size_t uid) { return *data_.at(uid); }
    /// @brief get the data of the func (by uid)
    auto &operator[](const expr &ex) { return get(ex.uid()); }
    const auto &get(size_t uid) const { return *data_.at(uid); }
    const auto &operator[](const expr &ex) const { return get(ex.uid()); }
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

constexpr inline auto format_as(approx_order order) { return magic_enum::enum_name<approx_order>(order); }
/////////////////////////////////////////////////////////////////////
/**
 * @brief sparse primal data
 * sparse view to access the input arguments of a function
 * @note it uses reference to avoid copying the input arguments
 */
struct func_arg_map {
    /**
     * @brief Construct a new sparse primal data object
     * @param primal sym data including states inputs etc
     * @param shared shared data
     * @param f function implementation pointer
     */
    func_arg_map(sym_data &primal, shared_data &shared, const generic_func &f);

    virtual ~func_arg_map() = default;
    const generic_func &func_;   ///< pointer to the func
    shared_data &shared_;        ///< ref to shared data
    sym_data *primal_ = nullptr; ///< reference to the primal data

    /**
     * @brief get the input argument values
     * @note this is a wrapper of in_args_ to access the values
     * @return vector_ref of input arguments
     */
    auto operator[](const sym &in) const {
        return in_args_[sym_uid_idx_.at(in.uid())];
    }
    /// @brief get the input argument values by index
    auto operator[](size_t i) const { return in_args_.at(i); }

    const auto &in_arg_data() const { return in_args_; }

    auto problem() const { return shared_.prob_; }

    // template <typename T>
    // T &as() { return dynamic_cast<T &>(*this); }

    template <typename T>
        requires(std::is_base_of_v<func_arg_map, T>)
    T &as() { return static_cast<T &>(*this); }

  protected:
    /// use ref to exploit sparsity (avoid copy)
    std::vector<vector_ref> in_args_;
    const std::unordered_map<size_t, size_t> &sym_uid_idx_;
};
/////////////////////////////////////////////////////////////////////
/**
 * @brief sparse approximation data
 * the dense data are mapped in the members of this class
 * @note in hess_ only the upper block triangular part are stored!(blocked by field)
 * for example Q_ux is store instead of Q_ux;
 */
struct func_approx_data : public func_arg_map {
    merit_data *merit_data_ = nullptr; ///< reference to the merit data
    ///////////////////////////////////////////////////
    vector_ref v_;                ///< value ref
    std::vector<matrix_ref> jac_; ///< jacobian references
    /// jacobian for cost, index corresponds to @ref func_arg_map::in_args_
    std::vector<row_vector_ref> merit_jac_;
    /// hessian for cost. index corresponds to @ref func_arg_map::in_args_
    std::vector<std::vector<matrix_ref>> merit_hess_;
    /**
     * @brief Construct a new sparse approx data object
     *
     * @param primal sym data including states inputs etc
     * @param raw dense raw data of approximation
     * @param shared shared data
     * @param f approximation
     */
    func_approx_data(sym_data &primal, merit_data &raw, shared_data &shared, const generic_func &f);
    /// @brief setup hessian from raw approx storage
    void setup_hessian();
    /// @brief get the jacobian reference
    auto jac(const sym &in) const { return jac_[sym_uid_idx_.at(in.uid())]; }
    auto jac(size_t i) const { return jac_.at(i); }

    virtual void reset() {}
};

def_unique_ptr(func_approx_data);
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