#ifndef __NODE_DATA__
#define __NODE_DATA__

#include <array>
#include <atri/core/offset_array.hpp>
#include <atri/ocp/core/approx.hpp>
#include <atri/ocp/core/sym_data.hpp>

namespace atri {

/**
 * @brief stacked approximation data
 * @note each std::vector contains the approximation data of functions in one field
 */

typedef offset_array<std::vector<sparse_approx_data_ptr_t>,
                     field::num_func, __dyn>
    stacked_approx_data;

struct node_data;
def_ptr(node_data);

/**
 * @brief node data class
 * stores the shooting node data including symbolics, raw approximation and its sparse mapping
 * @note to use your own data class with data_mgr, inherit this class and implement constructor C(problem_ptr_t)
 */
struct node_data {
    sym_data *sym_;
    approx_data *raw_;
    stacked_approx_data approx_;
    node_data(problem_ptr_t prob);
    ~node_data();
};
} // namespace atri

#endif