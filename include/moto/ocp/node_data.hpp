#ifndef __NODE_DATA__
#define __NODE_DATA__

#include <array>
#include <moto/core/array.hpp>
#include <moto/ocp/func.hpp>

namespace moto {

struct node_data;
def_unique_ptr(node_data);

/**
 * @brief node data class
 * stores the shooting node data including symbolics, raw approximation and its sparse mapping
 * @note to use your own data class with data_mgr, inherit this class and implement constructor C(ocp_ptr_t)
 */
struct node_data {
    ocp_ptr_t ocp_;              /// < pointer to the problem
    sym_data_ptr_t sym_;         /// < dense storage of symbolic data
    approx_storage_ptr_t dense_; /// <dense storage of the func data
    shared_data_ptr_t shared_;   /// < shared data
    shifted_array<std::vector<sp_approx_map_ptr_t>, field::num_func, __dyn>
        sparse_; /// < sparse view per func
    node_data(const ocp_ptr_t &prob);
    virtual ~node_data() = default;
    // get value of the whole field
    auto &value(field_t f) {
        if (f >= field::num_sym && f - __dyn <= field::num_constr)
            return dense_->approx_[f].v_;
        else
            return sym_->value_[f];
    }
    // get value of the sym variable
    auto value(const sym &sym) { return (*sym_)[sym]; }
    /**
     * @brief get the sparse func data by pointer
     *
     * @param f
     * @return auto&
     */
    auto &data(const func_impl &f) {
        return *sparse_[f.field_][ocp_->pos_by_uid_[f.uid_]];
    }
    /**
     * @brief get the sparse func data by pointer
     *
     * @param f
     * @return auto&
     */
    template <typename derived>
        requires std::is_base_of_v<func_impl, derived>
    auto &data(const std::shared_ptr<derived> &f) {
        return data(*f);
    }

    void update_approximation(bool eval_only = false);
};

} // namespace moto

#endif