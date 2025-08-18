#include <moto/spmm/sparse_mat.hpp>

namespace moto {
void sparse_mat::resize(size_t rows, size_t cols) {
    // check consistency
    for (const auto &panel : dense_panels_) {
        assert(panel.row_st_ + panel.rows_ < rows &&
               panel.col_st_ + panel.cols_ < cols &&
               "Dense panel size exceeds new matrix size");
    }
    for (const auto &panel : diag_panels_) {
        assert(panel.row_st_ + panel.rows_ < rows &&
               panel.col_st_ + panel.cols_ < cols &&
               "Diagonal panel size exceeds new matrix size");
    }
    for (const auto &panel : eye_panels_) {
        assert(panel.row_st_ + panel.rows_ < rows &&
               panel.col_st_ + panel.cols_ < cols &&
               "Eye panel size exceeds new matrix size");
    }
    rows_ = rows;
    cols_ = cols;
}
matrix_ref sparse_mat::insert(size_t r_st, size_t c_st, size_t r, size_t c, sparsity sp) {
    rows_ = std::max(rows_, r_st + r);
    cols_ = std::max(cols_, c_st + c);
    switch (sp) {
    case sparsity::dense:
        dense_panels_.emplace_back(r_st, c_st, r, c);
        return dense_panels_.back().mat();
    case sparsity::diag:
        diag_panels_.emplace_back(r_st, c_st, r, c);
        return diag_panels_.back().mat();
    case sparsity::eye:
        eye_panels_.emplace_back(r_st, c_st, r, c);
        return eye_panels_.back().mat();
    default:
        throw std::runtime_error("Unknown sparsity type");
    }
}
void sparse_mat::dump_into(matrix_ref out, dump_config cfg) const {
    if (cfg.overwrite) {
        if (cfg.add) {
            for (const auto &panel : dense_panels_) {
                out.block(panel.row_st_, panel.col_st_, panel.rows_, panel.cols_) = panel.data_;
            }
            for (const auto &panel : diag_panels_) {
                out.block(panel.row_st_, panel.col_st_, panel.rows_, panel.cols_).diagonal().array() = panel.data_.array();
            }
            for (const auto &panel : eye_panels_) {
                out.block(panel.row_st_, panel.col_st_, panel.rows_, panel.cols_).diagonal().array() = 1.0;
            }
        } else {
            for (const auto &panel : dense_panels_) {
                out.block(panel.row_st_, panel.col_st_, panel.rows_, panel.cols_) = -panel.data_;
            }
            for (const auto &panel : diag_panels_) {
                out.block(panel.row_st_, panel.col_st_, panel.rows_, panel.cols_).diagonal().array() = -panel.data_.array();
            }
            for (const auto &panel : eye_panels_) {
                out.block(panel.row_st_, panel.col_st_, panel.rows_, panel.cols_).diagonal().array() = -1.0;
            }
        }
    } else {
        if (cfg.add) {
            for (const auto &panel : dense_panels_) {
                out.block(panel.row_st_, panel.col_st_, panel.rows_, panel.cols_) += panel.data_;
            }
            for (const auto &panel : diag_panels_) {
                out.block(panel.row_st_, panel.col_st_, panel.rows_, panel.cols_).diagonal().array() += panel.data_.array();
            }
            for (const auto &panel : eye_panels_) {
                out.block(panel.row_st_, panel.col_st_, panel.rows_, panel.cols_).diagonal().array() += 1.0;
            }
        } else {
            for (const auto &panel : dense_panels_) {
                out.block(panel.row_st_, panel.col_st_, panel.rows_, panel.cols_) -= panel.data_;
            }
            for (const auto &panel : diag_panels_) {
                out.block(panel.row_st_, panel.col_st_, panel.rows_, panel.cols_).diagonal().array() -= panel.data_.array();
            }
            for (const auto &panel : eye_panels_) {
                out.block(panel.row_st_, panel.col_st_, panel.rows_, panel.cols_).diagonal().array() -= 1.0;
            }
        }
    }
}

matrix sparse_mat::dense() const {
    matrix out(rows_, cols_);
    out.setZero();
    dump_into(out);
    return out;
}
} // namespace moto