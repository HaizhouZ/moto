#include <moto/core/linear_backend.hpp>
#include <moto/core/sparse_matrix.hpp>

namespace moto {
void sparse_matrix::plan(std::span<const sparse_block_spec> blocks) {
  struct group {
    sparse_block_spec box;
    std::vector<size_t> blocks;
  };
  std::vector<group> groups;
  for (size_t i = 0; i < blocks.size(); ++i)
    groups.push_back({blocks[i], {i}});
  bool merged;
  do {
    merged = false;
    for (size_t i = 0; i < groups.size() && !merged; ++i)
      for (size_t j = i + 1; j < groups.size(); ++j) {
        auto &a = groups[i].box;
        const auto &b = groups[j].box;
        if (a.pattern != b.pattern) continue;
        const bool dense = a.pattern == sparsity::dense && a.row == b.row &&
                           a.rows == b.rows &&
                           (a.col + a.cols == b.col || b.col + b.cols == a.col);
        const bool diagonal = a.pattern != sparsity::dense &&
            ((a.row + a.rows == b.row && a.col + a.cols == b.col) ||
             (b.row + b.rows == a.row && b.col + b.cols == a.col));
        if (!dense && !diagonal) continue;
        const auto row_end = std::max(a.row + a.rows, b.row + b.rows);
        const auto col_end = std::max(a.col + a.cols, b.col + b.cols);
        a.row = std::min(a.row, b.row);
        a.col = std::min(a.col, b.col);
        a.rows = row_end - a.row;
        a.cols = col_end - a.col;
        groups[i].blocks.insert(groups[i].blocks.end(), groups[j].blocks.begin(),
                                groups[j].blocks.end());
        groups.erase(groups.begin() + j);
        merged = true;
        break;
      }
  } while (merged);

  for (const auto &group : groups) {
    const auto &g = group.box;
    size_t panel;
    if (g.pattern == sparsity::dense) {
      panel = dense_panels_.size();
      dense_panels_.emplace_back(g.row, g.col, g.rows, g.cols);
    } else if (g.pattern == sparsity::diag) {
      panel = diag_panels_.size();
      diag_panels_.emplace_back(g.row, g.col, g.rows, g.cols);
    } else {
      panel = eye_panels_.size();
      eye_panels_.emplace_back(g.row, g.col, g.rows, g.cols);
    }
    for (const auto bi : group.blocks)
      planned_.push_back({blocks[bi], panel, blocks[bi].row - g.row,
                          blocks[bi].col - g.col});
  }
}

void sparse_matrix::resize(size_t rows, size_t cols) {
  jit_cache_.reset();
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
bool sparse_matrix::valid() const {
  for (const auto &panel : dense_panels_)
    if (panel.data_.hasNaN() || panel.data_.allFinite() == false)
      return false;
  for (const auto &panel : diag_panels_)
    if (panel.data_.hasNaN() || panel.data_.allFinite() == false)
      return false;
  return true;
}
bool sparse_matrix::set_dynamic_eye(bool enabled) {
  if (dynamic_eye_ == enabled)
    return false;
  dynamic_eye_ = enabled;
  jit_cache_.reset();
  return true;
}
sparse_matrix &sparse_matrix::operator=(const sparse_matrix &other) {
  if (this != &other)
    jit_cache_.reset();
  if (this != &other && this->is_empty()) {
    rows_ = other.rows_;
    cols_ = other.cols_;
    dense_panels_ = other.dense_panels_;
    diag_panels_ = other.diag_panels_;
    eye_panels_ = other.eye_panels_;
    dynamic_eye_ = other.dynamic_eye_;
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
    for (size_t i = 0; i < other.eye_panels_.size(); i++) {
      eye_panels_[i].data_ = other.eye_panels_[i].data_;
    }
    dynamic_eye_ = other.dynamic_eye_;
  }
  return *this;
}
void sparse_matrix::setZero() {
  for (auto &panel : dense_panels_) {
    panel.data_.setZero();
  }
  for (auto &panel : diag_panels_) {
    panel.data_.setZero();
  }
  // eye panels do not need to be set to zero
}
matrix_ref sparse_matrix::insert(size_t r_st, size_t c_st, size_t r, size_t c,
                                 sparsity sp) {
  jit_cache_.reset();
  static matrix empty;
  static vector empty_vec;
  static row_vector empty_rvec;
  if (r == 0 || c == 0)
    if (c == 1 || sp == sparsity::eye || sp == sparsity::diag) {
      return empty_vec;
    } else if (r == 1) {
      return empty_rvec;
    } else {
      return empty;
    }
  assert(r_st + r <= rows_ && c_st + c <= cols_ &&
         "Inserted panel exceeds matrix size");
  const sparse_block_spec requested{r_st, c_st, r, c, sp};
  for (auto &binding : planned_) {
    if (binding.used || binding.block != requested) continue;
    binding.used = true;
    if (sp == sparsity::dense)
      return dense_panels_[binding.panel].data_.block(
          binding.local_row, binding.local_col, r, c);
    if (sp == sparsity::diag)
      return diag_panels_[binding.panel].data_.segment(binding.local_row, r);
    return eye_panels_[binding.panel].data_.segment(binding.local_row, r);
  }
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
matrix sparse_matrix::dense() const {
  matrix out(rows_, cols_);
  out.setZero();
  linear_backend::run_dense_write(*this, out.data(), out.rows(), 1., false);
  return out;
}
} // namespace moto
