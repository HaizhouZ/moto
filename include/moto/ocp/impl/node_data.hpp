#ifndef __MOTO_NODE_DATA_HPP__
#define __MOTO_NODE_DATA_HPP__

#include <moto/ocp/impl/func.hpp>

namespace moto {
struct node_data;
def_unique_ptr(node_data);

/**
 * @brief node data class
 * stores the shooting node data including symbolics, raw approximation and its sparse mapping
 * @note to use your own data class with data_mgr, inherit this class and implement constructor C(ocp_ptr_t)
 */
struct node_data {
    ocp_ptr_t prob_;             /// < pointer to the problem
    sym_data_ptr_t sym_;         /// < dense storage of symbolic data
    approx_storage_ptr_t dense_; /// <dense storage of the func data
    shared_data_ptr_t shared_;   /// < shared data
    shifted_array<std::vector<sp_approx_map_ptr_t>, field::num_func, __dyn>
        sparse_; /// < sparse view per func
    node_data(const ocp_ptr_t &prob);
    virtual ~node_data() = default;
    // get value of the whole field
    auto &value(field_t f) const {
        if (f >= field::num_sym && f - __dyn <= field::num_constr)
            return dense_->approx_[f].v_;
        else
            return sym_->value_[f];
    }
    // get value of the sym variable
    auto value(const sym &sym) const { return (*sym_)[sym]; }
    /**
     * @brief get the sparse func data by pointer
     *
     * @param f
     * @return auto&
     */
    auto &data(const impl::func &f) const {
        return *sparse_[f.field_][prob_->pos_by_uid_[f.uid_]];
    }
    /**
     * @brief get the sparse func data by pointer
     *
     * @param f
     * @return auto&
     */
    template <typename derived>
        requires std::is_base_of_v<impl::func, derived>
    auto &data(const std::shared_ptr<derived> &f) const {
        return data(*f);
    }

    scalar_t cost() const { return dense_->cost_; }

    scalar_t inf_prim_res() const; // constraint violation residual

    void update_approximation(bool eval_only = false);
};
} // namespace moto

#endif