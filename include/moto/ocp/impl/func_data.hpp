#ifndef MOTO_OCP_IMPL_FUNC_DATA_HPP
#define MOTO_OCP_IMPL_FUNC_DATA_HPP

#include <moto/ocp/impl/lag_data.hpp>
#include <moto/ocp/impl/sym_data.hpp>

namespace moto {
struct func_arg_map;
def_unique_ptr(func_arg_map);
class generic_func;
class shared_data {
    std::unordered_map<size_t, func_arg_map_ptr_t> data_;

  public:
    shared_data(ocp *prob, sym_data *primal, lag_data *raw = nullptr);
    shared_data(shared_data &&) = default;

    const ocp *prob_;
    void add(size_t uid, func_arg_map_ptr_t &&data);
    template <typename derived>
        requires std::is_base_of_v<generic_func, derived>
    void add(const derived &ex, func_arg_map_ptr_t &&data) {
        assert(ex.field() == __pre_comp || ex.field() == __usr_func);
        add(ex.uid(), std::move(data));
    }
    func_arg_map *try_get(size_t uid);
    func_arg_map &get(size_t uid);
    func_arg_map &operator[](const expr &ex);
    const func_arg_map &get(size_t uid) const;
    const func_arg_map &operator[](const expr &ex) const;
};
def_unique_ptr(shared_data);
/////////////////////////////////////////////////////////////////////

enum class approx_order { none = 0,
                          zero,
                          first,
                          second };

constexpr inline auto format_as(approx_order order) { return magic_enum::enum_name<approx_order>(order); }
/////////////////////////////////////////////////////////////////////
struct func_arg_map {
    func_arg_map(sym_data &primal, shared_data &shared, const generic_func &f);

    virtual ~func_arg_map() = default;
    const generic_func &func_;
    shared_data &shared_;
    sym_data *primal_ = nullptr;

    vector_ref operator[](const sym &in) const;
    vector_ref operator[](size_t i) const;

    const std::vector<vector_ref> &in_arg_data() const;

    const ocp *problem() const;

    template <typename T>
        requires(std::is_base_of_v<func_arg_map, T>)
    T &as() { return static_cast<T &>(*this); }

  protected:
    std::vector<vector_ref> in_args_;
};
/////////////////////////////////////////////////////////////////////
struct func_approx_data : public func_arg_map {
    lag_data *lag_data_ = nullptr;
    ///////////////////////////////////////////////////
    vector_ref v_;
    std::vector<matrix_ref> jac_;
    std::vector<std::vector<matrix_ref>> lag_hess_;
    func_approx_data(sym_data &primal, lag_data &raw, shared_data &shared, const generic_func &f);
    void setup_hessian();
    bool has_jacobian_block(size_t arg_idx) const;
    matrix_ref jac(const sym &in) const;
    matrix_ref jac(size_t i) const;
};

def_unique_ptr(func_approx_data);
/////////////////////////////////////////////////////////////////////
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
