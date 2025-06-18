#ifndef __NODE_DATA__
#define __NODE_DATA__

#include <array>
#include <atri/core/array.hpp>
#include <atri/ocp/func.hpp>

namespace atri {

struct node_data;
def_unique_ptr(node_data);

/**
 * @brief node data class
 * stores the shooting node data including symbolics, raw approximation and its sparse mapping
 * @note to use your own data class with data_mgr, inherit this class and implement constructor C(problem_ptr_t)
 */
struct node_data {
    sym_data_ptr_t sym_;         /// < dense storage of symbolic data
    approx_storage_ptr_t dense_; /// <dense storage of the func data
    shared_data_ptr_t shared_;   /// < shared data
    shifted_array<std::vector<sparse_approx_data_ptr_t>, field::num_func, __dyn>
        sparse_;                                     /// < sparse view per func
    std::vector<sparse_primal_data_ptr_t> usr_data_; /// < user defined data
    node_data(problem_ptr_t prob);
};

} // namespace atri

#endif