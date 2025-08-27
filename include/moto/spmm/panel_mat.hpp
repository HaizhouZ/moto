#ifndef MOTO_SPMM_PANEL_MAT_HPP
#define MOTO_SPMM_PANEL_MAT_HPP

#include <moto/spmm/fwd.hpp>

namespace moto {


// Forward declaration for panel_mat with sparsity as non-type template parameter
template <sparsity Sp>
struct panel_mat;

struct panel_mat_base {
    int row_st_;
    int col_st_;
    int row_ed_;
    int col_ed_;
    int rows_;
    int cols_;
    sparsity sp_;
    panel_mat_base(size_t r_st, size_t c_st, size_t r, size_t c, sparsity sp)
        : row_st_(r_st), col_st_(c_st), rows_(r), cols_(c), sp_(sp) {
        row_ed_ = row_st_ + rows_;
        col_ed_ = col_st_ + cols_;
    }
    virtual scalar_t *data() = 0;
    template <sparsity Sp>
    panel_mat<Sp> &as() { return static_cast<panel_mat<Sp> &>(*this); }
    void copy_dim(const panel_mat_base &other) {
        row_st_ = other.row_st_;
        col_st_ = other.col_st_;
        row_ed_ = other.row_ed_;
        col_ed_ = other.col_ed_;
        rows_ = other.rows_;
        cols_ = other.cols_;
    }
    virtual matrix_ref mat() = 0;
};

template <sparsity Sp = sparsity::dense>
struct panel_mat : public panel_mat_base {
    static constexpr auto sparsity_type = Sp;
    panel_mat(size_t r_st, size_t c_st, size_t r, size_t c, bool setup = true)
        : panel_mat_base(r_st, c_st, r, c, Sp) {
        if (setup)
            setup_data();
    }
    std::conditional_t<Sp == sparsity::dense, matrix, vector> data_;

    void setup_data() {
        if constexpr (Sp == sparsity::dense) {
            data_ = matrix::Zero(rows_, cols_);
        } else if constexpr (Sp == sparsity::diag) {
            data_ = vector::Zero(rows_);
        } else if constexpr (Sp == sparsity::eye) {
            data_ = vector::Ones(rows_); // eye matrix is represented as a vector of ones
        }
    }

    scalar_t *data() final override { return data_.data(); }

    matrix_ref mat() final override { return data_; }
};
} // namespace moto
#endif