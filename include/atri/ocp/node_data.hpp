#ifndef __NODE_DATA__
#define __NODE_DATA__

#include <array>
#include <atri/core/array.hpp>
#include <atri/ocp/approx.hpp>
#include <atri/ocp/sym_data.hpp>

namespace atri {

struct node_data;
def_unique_ptr(node_data);

/**
 * @brief node data class
 * stores the shooting node data including symbolics, raw approximation and its sparse mapping
 * @note to use your own data class with data_mgr, inherit this class and implement constructor C(problem_ptr_t)
 */
struct node_data {
    sym_data *sym_;       /// < dense storage of symbolic data
    approx_storage *raw_; /// <dense storage of the approx data
    shifted_array<std::vector<sparse_approx_data_ptr_t>, field::num_func, __dyn>
        sparse_; /// < sparse view per approx
    node_data(problem_ptr_t prob);
    ~node_data();
};
} // namespace atri

#endif