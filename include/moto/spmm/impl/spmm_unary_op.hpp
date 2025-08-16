#ifndef MOTO_SPMM_IMPL_SPMM_UNARY_OP_HPP
#define MOTO_SPMM_IMPL_SPMM_UNARY_OP_HPP

namespace moto {
    
enum sparsity;
struct sparse_mat;

namespace spmm {
template <sparsity Sp, typename Callback>
auto unary_select(const sparse_mat &s, Callback &&callback) {
    if constexpr (Sp == sparsity::dense) {
        for (const auto &panel : s.dense_panels_) {
            callback(panel);
        }
    } else if constexpr (Sp == sparsity::diag) {
        for (const auto &panel : s.diag_panels_) {
            callback(panel);
        }
    } else if constexpr (Sp == sparsity::eye) {
        for (const auto &panel : s.eye_panels_) {
            callback(panel);
        }
    } else {
    }
};
} // namespace spmm
} // namespace moto

#endif