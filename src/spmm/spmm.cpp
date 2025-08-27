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
bool sparse_mat::valid() const {
    for (const auto &panel : dense_panels_)
        if (panel.data_.hasNaN() || panel.data_.allFinite() == false)
            return false;
    for (const auto &panel : diag_panels_)
        if (panel.data_.hasNaN() || panel.data_.allFinite() == false)
            return false;
    return true;
}
sparse_mat &sparse_mat::operator=(const sparse_mat &other) {
    if (this != &other && this->is_empty()) {
        rows_ = other.rows_;
        cols_ = other.cols_;
        dense_panels_ = other.dense_panels_;
        diag_panels_ = other.diag_panels_;
        eye_panels_ = other.eye_panels_;
    } else {
        assert(rows_ == other.rows_ && cols_ == other.cols_);
        assert(dense_panels_.size() == other.dense_panels_.size());
        assert(diag_panels_.size() == other.diag_panels_.size());
        assert(eye_panels_.size() == other.eye_panels_.size());
        for (size_t i = 0; i < other.dense_panels_.size(); i++) {
            dense_panels_[i].data_ = other.dense_panels_[i].data_;
        }
        for (size_t i = 0; i < other.diag_panels_.size(); i++) {
            diag_panels_[i].data_ = other.diag_panels_[i].data_;
        }
    }
    return *this;
}
void sparse_mat::setZero() {
    for (auto &panel : dense_panels_) {
        panel.data_.setZero();
    }
    for (auto &panel : diag_panels_) {
        panel.data_.setZero();
    }
    // eye panels do not need to be set to zero
}
matrix_ref sparse_mat::insert(size_t r_st, size_t c_st, size_t r, size_t c, sparsity sp) {
    rows_ = std::max(rows_, r_st + r);
    cols_ = std::max(cols_, c_st + c);
    switch (sp) {
    case sparsity::dense:
        dense_panels_.emplace_back(r_st, c_st, r, c);
        return dense_panels_.back().mat();
    case sparsity::diag:
        assert(r == c && "Diagonal panel must be square");
        diag_panels_.emplace_back(r_st, c_st, r, c);
        return diag_panels_.back().mat();
    case sparsity::eye:
        assert(r == c && "Eye panel must be square");
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