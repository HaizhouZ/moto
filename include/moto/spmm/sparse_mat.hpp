#ifndef MOTO_SPMM_SPARSE_MAT_HPP
#define MOTO_SPMM_SPARSE_MAT_HPP

#include <moto/spmm/panel_mat.hpp>

namespace moto {

struct sparse_mat {
    size_t rows_ = 0;
    size_t cols_ = 0;
    std::vector<panel_mat<sparsity::dense>> dense_panels_;
    std::vector<panel_mat<sparsity::diag>> diag_panels_;
    std::vector<panel_mat<sparsity::eye>> eye_panels_;
    sparse_mat() = default;
    matrix_ref insert(size_t r_st, size_t c_st, size_t r, size_t c, sparsity sp);
    void inner_product(const matrix &m, matrix &out);
    template <typename rhs_type, typename out_type, bool add = true>
    void times(const rhs_type &rhs, out_type &out);
    template <typename rhs_type, typename out_type, bool add = true>
    void T_times(const rhs_type &rhs, out_type &out);
    template <typename lhs_type, typename out_type, bool add = true>
    void right_times(const lhs_type &lhs, out_type &out);
    template <typename lhs_type, typename out_type, bool add = true>
    void right_T_times(const lhs_type &lhs, out_type &out);
    void dump_into(matrix_ref out) const;
    matrix dense() const;
    template <typename rhs_type>
    sparse_mat &operator=(const rhs_type &rhs) {
        for (auto &panel : dense_panels_) {
            if (panel.rows_ == rhs.rows())
                panel.data_.noalias() = rhs.middleCols(panel.col_st_, panel.cols_);
        }
        return *this;
    }
};

} // namespace moto
#endif