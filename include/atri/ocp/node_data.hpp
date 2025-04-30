#ifndef __NODE_DATA__
#define __NODE_DATA__

#include <array>

#include <atri/ocp/approx.hpp>

namespace atri {

/**
 * @brief stacked approximation data
 * @note each std::vector contains the approximation data of functions in one
 * field
 */
class approx_sets_data {
    std::array<std::vector<sparse_approx_data_ptr_t>, field::num_func> ptr_;

  public:
    std::vector<sparse_approx_data_ptr_t> &operator[](size_t idx) {
        return ptr_[idx - field::num_sym];
    }
    const std::vector<sparse_approx_data_ptr_t> &operator[](size_t idx) const {
        return ptr_[idx - field::num_sym];
    }
    void swap(approx_sets_data &rhs) { ptr_.swap(rhs.ptr_); }
};

struct node_data;
def_ptr(node_data);

struct node_data {
    raw_data raw_data_;
    approx_sets_data approx_;
    node_data(approx_sets_data &&a, raw_data &&p)
        : approx_(std::move(a)), raw_data_(std::move(p)) {}
    node_data(problem_ptr_t exprs);
};
} // namespace atri

#endif