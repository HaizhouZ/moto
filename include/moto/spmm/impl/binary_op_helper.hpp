#ifndef MOTO_SPMM_IMPL_BINARY_OP_HELPER_HPP
#define MOTO_SPMM_IMPL_BINARY_OP_HELPER_HPP

namespace moto {
struct sparse_mat;
namespace spmm {
enum binary_op_type {
    times,
    T_times,
    right_times,
    right_T_times
};

template <typename other_t, typename out_type>
struct is_consistent {
    constexpr static bool other_row_fixed = other_t::RowsAtCompileTime != -1;
    constexpr static size_t other_row = other_row_fixed ? other_t::RowsAtCompileTime : 0;
    constexpr static bool out_row_fixed = out_type::RowsAtCompileTime != -1;
    constexpr static size_t out_row = out_row_fixed ? out_type::RowsAtCompileTime : 0;
    constexpr static bool other_col_fixed = other_t::ColsAtCompileTime != -1;
    constexpr static size_t other_col = other_col_fixed ? other_t::ColsAtCompileTime : 0;
    constexpr static bool out_col_fixed = out_type::ColsAtCompileTime != -1;
    constexpr static size_t out_col = out_col_fixed ? out_type::ColsAtCompileTime : 0;

    static constexpr bool value(binary_op_type op) {
        switch (op) {
        case binary_op_type::times:
            return !((other_row_fixed && other_row == 1) || (out_row_fixed && out_row == 1));
        case binary_op_type::T_times:
            return !((other_row_fixed && other_row == 1) || (out_row_fixed && out_row == 1));
        case binary_op_type::right_times:
            return !((other_col_fixed && other_col == 1) || (out_col_fixed && out_col == 1));
        case binary_op_type::right_T_times:
            return !((other_row_fixed && other_row == 1) || (out_col_fixed && out_col == 1));
        default:
            return false;
        }
    }
};
template <typename out_type>
struct is_consistent<sparse_mat, out_type> {
    static constexpr bool value(binary_op_type) { return true; }
};
} // namespace spmm
} // namespace moto

#endif