#ifndef __MOTO_NODE_DATA_HPP__
#define __MOTO_NODE_DATA_HPP__

#include <moto/ocp/impl/func.hpp>
#include <moto/ocp/problem.hpp>

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
struct node_data {
    scalar_t inf_prim_res_ = 0.;
    scalar_t inf_comp_res_ = 0.;

  protected:
    ocp_ptr_t prob_;                /// < pointer to the problem
    sym_data_ptr_t sym_;            /// < dense storage of symbolic data
    merit_data_ptr_t dense_; /// <dense storage of the func data
    shared_data_ptr_t shared_;      /// < shared data
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
    auto &shared() const { return *shared_; } ///< getter for impl_
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
    auto &data(const func &f) const { return *sparse_[f->field()][prob_->pos(f)]; }

    scalar_t cost() const { return dense_->cost_; }

    /**
     * @brief update the approximation data and compute primal and comp residuals
     *
     * @param eval_only
     */
    void update_approximation(bool eval_only = false);

    template <std::array fields, typename Callback>
        requires std::is_invocable_r_v<void, Callback, const generic_func &, func_approx_data &> &&
                 std::is_same_v<std::tuple_element_t<0, decltype(fields)>, field_t>
    void for_each(Callback &&callback) {
        for (const auto &field : fields) {
            size_t idx = 0;
            auto &s = this->sparse_[field];
            for (const generic_func &f : prob_->exprs(field)) {
                callback(f, *s[idx]);
                idx++;
            }
        }
    }

    template <typename Callback>
    void for_each_constr(Callback &&f) {
        for_each<constr_fields>(std::forward<Callback>(f));
    }

    void clear_merit_jac();
    void clear_merit_hessian();
};
} // namespace moto

#endif