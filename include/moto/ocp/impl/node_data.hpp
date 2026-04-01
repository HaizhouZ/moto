#ifndef __MOTO_NODE_DATA_HPP__
#define __MOTO_NODE_DATA_HPP__

#include <moto/ocp/impl/custom_func.hpp>
#include <moto/ocp/problem.hpp>
#include <moto/utils/func_traits.hpp>

namespace moto {
struct node_data;
def_unique_ptr(node_data);
namespace impl {
class data_mgr;
} // namespace impl
/**
 * @brief node data class
 * stores the shooting node data including symbolics, raw approximation and its sparse mapping
 * @note to use your own data class with data_mgr, inherit this class and implement constructor C(ocp_ptr_t)
 */
struct MOTO_ALIGN_NO_SHARING node_data {
    scalar_t inf_prim_res_ = 0.;
    scalar_t prim_res_l1_ = 0.;
    scalar_t inf_comp_res_ = 0.;

  protected:
    ocp_ptr_t prob_;           /// < pointer to the problem
    sym_data_ptr_t sym_;       /// < dense storage of symbolic data
    lag_data_ptr_t dense_;     /// <dense storage of the func data
    shared_data_ptr_t shared_; /// < shared data
    shifted_array<std::vector<func_approx_data_ptr_t>, field::num_func, __dyn>
        sparse_; /// < sparse view per func

    friend class impl::data_mgr; ///< data manager can access private members
  public:
    node_data(const ocp_ptr_t &prob);
    node_data(const node_data &rhs) = delete;
    node_data(node_data &&rhs) noexcept = default;
    virtual ~node_data() = default;

    auto &sym_val() const { return *sym_; }   ///< getter for sym_
    auto &dense() const { return *dense_; }   ///< getter for dense_
    auto &shared() const { return *shared_; } ///< getter for shared_
    auto &problem() const { return *prob_; }  ///< getter for prob_

    // get value of the whole field
    auto &value(field_t f) const {
        if (f >= field::num_sym && f - __dyn <= field::num_constr)
            return dense_->approx_[f].v_;
        else
            return sym_->value_[f];
    }
    auto value(const sym &s) const { return (*sym_)[s]; }
    /**
     * @brief get the sparse func data by pointer
     *
     * @param f
     * @return auto&
     */
    auto &data(const func &f) const {
        if (in_field(f->field(), custom_func_fields))
            throw std::runtime_error("custom_func does not have sparse data");
        return *sparse_[f->field()][prob_->pos(f)];
    }

    auto &data(const custom_func &f) const { return shared_->get(f->uid()); }

    scalar_t cost() const { return dense_->cost_; }

    enum class update_mode : size_t {
        eval_val = 0,     ///< evaluate value
        eval_jac,         ///< evaluate jacobian
        eval_hess,        ///< evaluate hessian
        eval_derivatives, ///< evaluate derivatives (jacobian, hessian)
        eval_all,         ///< evaluate all (value, jacobian, hessian)
    };
    /**
     * @brief update the approximation data and compute primal and comp residuals
     *
     * @param eval_only
     */
    void update_approximation(update_mode config = update_mode::eval_all,
                              bool include_original_cost = true);

    template <typename Callback>
    void for_each(field_t field, Callback &&callback) {
        using func_info = utils::func_traits<Callback>;
        static_assert(func_info::arg_num == 2 && "wrong number of arguments for callback");
        using func_type = std::decay_t<typename func_info::arg_type<0>>;
        using approx_type = std::decay_t<typename func_info::arg_type<1>>;
        static_assert(std::is_base_of_v<generic_func, func_type> && std::is_base_of_v<func_approx_data, approx_type>,
                      "Callback must accept a (derived) generic_func and (derived) func_approx_data");
        size_t idx = 0;
        auto &s = this->sparse_[field];
        for (const func_type &f : prob_->exprs(field)) {
            callback(f, s[idx]->as<approx_type>());
            idx++;
        }
    }

    template <std::array fields, typename Callback>
    void for_each(Callback &&callback) {
        for (const field_t &field : fields) {
            for_each(field, std::forward<Callback>(callback));
        }
    }

    template <typename Callback>
    void for_each_constr(Callback &&f) {
        for_each<constr_fields>(std::forward<Callback>(f));
    }

    void clear_lag_jac();
    void clear_lag_hessian();

    void print_residuals() const;
};
} // namespace moto

#endif
