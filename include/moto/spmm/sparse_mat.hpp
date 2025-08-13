#ifndef MOTO_SPMM_SPARSE_MAT_HPP
#define MOTO_SPMM_SPARSE_MAT_HPP

#include <moto/spmm/panel_mat.hpp>

namespace moto {

struct sparse_mat {
    size_t rows_ = 0;
    size_t cols_ = 0;
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    std::vector<panel_mat<sparsity::dense>> dense_panels_;
    std::vector<panel_mat<sparsity::diag>> diag_panels_;
    std::vector<panel_mat<sparsity::eye>> eye_panels_;
    sparse_mat() = default;
    void resize(size_t rows, size_t cols);
    matrix_ref insert(size_t r_st, size_t c_st, size_t r, size_t c, sparsity sp);
    void inner_product(const matrix &m, matrix &out);
    template <bool add = true, typename rhs_type, typename out_type>
    void times(const rhs_type &rhs, out_type &out);
    template <bool add = true, typename rhs_type, typename out_type>
    void T_times(const rhs_type &rhs, out_type &out);
    template <bool add = true, typename lhs_type, typename out_type>
    void right_times(const lhs_type &lhs, out_type &out);
    template <bool add = true, typename lhs_type, typename out_type>
    void right_T_times(const lhs_type &lhs, out_type &out);
    void dump_into(matrix_ref out, bool add = true) const;
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

#define EXPORT_EXPLICIT_SP_MEMFUNC_INSTANTIATE_IMPL(func, lhs_type, rhs_type)                            \
    extern template void sparse_mat::func<true, lhs_type, rhs_type>(const lhs_type &rhs, rhs_type &out); \
    extern template void sparse_mat::func<false, lhs_type, rhs_type>(const lhs_type &rhs, rhs_type &out);

#define EXPORT_EXPLICIT_SP_MEMFUNC_INSTANTIATE(func)                           \
    EXPORT_EXPLICIT_SP_MEMFUNC_INSTANTIATE_IMPL(func, sparse_mat, matrix);     \
    EXPORT_EXPLICIT_SP_MEMFUNC_INSTANTIATE_IMPL(func, sparse_mat, vector);     \
    EXPORT_EXPLICIT_SP_MEMFUNC_INSTANTIATE_IMPL(func, matrix, matrix);         \
    EXPORT_EXPLICIT_SP_MEMFUNC_INSTANTIATE_IMPL(func, matrix, vector);         \
    EXPORT_EXPLICIT_SP_MEMFUNC_INSTANTIATE_IMPL(func, row_vector, row_vector); \
    EXPORT_EXPLICIT_SP_MEMFUNC_INSTANTIATE_IMPL(func, row_vector, vector);     \
    EXPORT_EXPLICIT_SP_MEMFUNC_INSTANTIATE_IMPL(func, vector, row_vector);     \
    EXPORT_EXPLICIT_SP_MEMFUNC_INSTANTIATE_IMPL(func, vector, vector);

EXPORT_EXPLICIT_SP_MEMFUNC_INSTANTIATE(times)
EXPORT_EXPLICIT_SP_MEMFUNC_INSTANTIATE(T_times)
EXPORT_EXPLICIT_SP_MEMFUNC_INSTANTIATE(right_times)
EXPORT_EXPLICIT_SP_MEMFUNC_INSTANTIATE(right_T_times)

} // namespace moto
#endif