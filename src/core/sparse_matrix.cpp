#include <moto/core/linear_backend.hpp>
#include <moto/core/sparse_matrix.hpp>


namespace moto {
sparse_layout_plan
make_sparse_layout_plan(std::span<const sparse_block_spec> blocks,
                        sparse_plan_mode mode) {
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
        const bool exact = a.row == b.row && a.col == b.col &&
                           a.rows == b.rows && a.cols == b.cols;
        const bool additive_dense =
            mode == sparse_plan_mode::additive && exact &&
            a.pattern == sparsity::dense && b.pattern == sparsity::dense;
        const auto structured = [](sparsity sp) {
          return sp == sparsity::diag || sp == sparsity::eye;
        };
        const bool additive_structured =
            mode == sparse_plan_mode::additive && exact &&
            structured(a.pattern) && structured(b.pattern);
        if (additive_structured)
          a.pattern = sparsity::diag;
        if (a.pattern != b.pattern && !additive_structured) continue;
        const bool dense = a.pattern == sparsity::dense && a.row == b.row &&
                           a.rows == b.rows &&
                           (a.col + a.cols == b.col || b.col + b.cols == a.col);
        const bool diagonal = a.pattern != sparsity::dense &&
            ((a.row + a.rows == b.row && a.col + a.cols == b.col) ||
             (b.row + b.rows == a.row && b.col + b.cols == a.col));
        if (!additive_dense && !additive_structured && !dense && !diagonal)
          continue;
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

  sparse_layout_plan result;
  size_t dense_panel = 0, diag_panel = 0, eye_panel = 0;
  for (const auto &group : groups) {
    const auto &g = group.box;
    size_t panel;
    if (g.pattern == sparsity::dense)
      panel = dense_panel++;
    else if (g.pattern == sparsity::diag)
      panel = diag_panel++;
    else if (g.pattern == sparsity::eye)
      panel = eye_panel++;
    else
      throw std::invalid_argument("Cannot plan unknown sparse panel");
    result.panels.push_back(g);
    for (const auto bi : group.blocks)
      result.bindings.push_back({blocks[bi], g.pattern, panel,
                                 blocks[bi].row - g.row,
                                 blocks[bi].col - g.col, bi});
  }
  return result;
}

void sparse_matrix::plan(const sparse_layout_plan &layout) {
  if (!is_empty() || !planned_.empty())
    throw std::logic_error("Sparse matrix layout can only be planned once");
  packed_diagonal_storage_ = layout.pack_diagonal_storage;
  constexpr size_t diagonal_alignment =
      std::max<size_t>(1, EIGEN_MAX_ALIGN_BYTES / sizeof(scalar_t));
  const auto align_diagonal = [=](size_t offset) {
    return (offset + diagonal_alignment - 1) / diagonal_alignment *
           diagonal_alignment;
  };
  size_t packed_diagonal_size = 0;
  if (packed_diagonal_storage_)
    for (const auto &p : layout.panels)
      if (p.pattern == sparsity::diag) {
        packed_diagonal_size = align_diagonal(packed_diagonal_size);
        packed_diagonal_size += p.rows;
      }
  if (packed_diagonal_size)
    diag_panels_.emplace_back(0, 0, packed_diagonal_size,
                              packed_diagonal_size);
  size_t diagonal_index = 0;
  size_t diagonal_offset = 0;
  for (const auto &p : layout.panels) {
    if (p.pattern == sparsity::dense)
      dense_panels_.emplace_back(p.row, p.col, p.rows, p.cols);
    else if (p.pattern == sparsity::diag) {
      if (packed_diagonal_storage_) {
        diagonal_offset = align_diagonal(diagonal_offset);
        diagonal_segments_.push_back({p.row, p.col, p.rows, p.cols, 0,
                                      diagonal_offset});
        diagonal_offset += p.rows;
      } else {
        diag_panels_.emplace_back(p.row, p.col, p.rows, p.cols);
        diagonal_segments_.push_back(
            {p.row, p.col, p.rows, p.cols, diagonal_index++, 0});
      }
    }
    else if (p.pattern == sparsity::eye)
      eye_panels_.emplace_back(p.row, p.col, p.rows, p.cols);
    else
      throw std::invalid_argument("Cannot allocate unknown sparse panel");
  }
  planned_.reserve(layout.bindings.size());
  for (const auto &b : layout.bindings)
    planned_.push_back(
        {b.block, b.storage_pattern, b.panel, b.local_row, b.local_col});
}

void sparse_matrix::plan(std::span<const sparse_block_spec> blocks,
                         sparse_plan_mode mode) {
  plan(make_sparse_layout_plan(blocks, mode));
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
    diagonal_segments_ = other.diagonal_segments_;
    packed_diagonal_storage_ = other.packed_diagonal_storage_;
    dynamic_eye_ = other.dynamic_eye_;
  } else {
    assert(rows_ == other.rows_ && cols_ == other.cols_);
    assert(dense_panels_.size() == other.dense_panels_.size());
    assert(diag_panels_.size() == other.diag_panels_.size());
    assert(eye_panels_.size() == other.eye_panels_.size());
    assert(diagonal_segments_.size() == other.diagonal_segments_.size());
    assert(packed_diagonal_storage_ == other.packed_diagonal_storage_);
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
    if (binding.storage_pattern == sparsity::dense)
      return dense_panels_[binding.panel].data_.block(
          binding.local_row, binding.local_col, r, c);
    if (binding.storage_pattern == sparsity::diag)
      return [&]() -> matrix_ref {
        const auto &segment = diagonal_segments_[binding.panel];
        return diag_panels_[segment.storage_panel].data_.segment(
            segment.storage_offset + binding.local_row, r);
      }();
    return eye_panels_[binding.panel].data_.segment(binding.local_row, r);
  }
  switch (sp) {
  case sparsity::dense:
    dense_panels_.emplace_back(r_st, c_st, r, c);
    return dense_panels_.back().mat();
  case sparsity::diag:
    assert(r == c && "Diagonal panel must be square");
    diag_panels_.emplace_back(r_st, c_st, r, c);
    diagonal_segments_.push_back(
        {r_st, c_st, r, c, diag_panels_.size() - 1, 0});
    return diag_panels_.back().mat();
  case sparsity::eye:
    assert(r == c && "Eye panel must be square");
    eye_panels_.emplace_back(r_st, c_st, r, c);
    return eye_panels_.back().mat();
  default:
    throw std::runtime_error("Unknown sparsity type");
  }
}
matrix_ref sparse_matrix::bind(size_t r_st, size_t c_st, size_t r, size_t c,
                               sparsity sp) {
  static matrix empty;
  static vector empty_vec;
  static row_vector empty_rvec;
  if (r == 0 || c == 0)
    if (c == 1 || sp == sparsity::eye || sp == sparsity::diag)
      return empty_vec;
    else if (r == 1)
      return empty_rvec;
    else
      return empty;
  const sparse_block_spec requested{r_st, c_st, r, c, sp};
  for (auto &binding : planned_) {
    if (binding.used || binding.block != requested) continue;
    binding.used = true;
    if (binding.storage_pattern == sparsity::dense)
      return dense_panels_[binding.panel].data_.block(
          binding.local_row, binding.local_col, r, c);
    if (binding.storage_pattern == sparsity::diag)
      return [&]() -> matrix_ref {
        const auto &segment = diagonal_segments_[binding.panel];
        return diag_panels_[segment.storage_panel].data_.segment(
            segment.storage_offset + binding.local_row, r);
      }();
    return eye_panels_[binding.panel].data_.segment(binding.local_row, r);
  }
  throw std::logic_error("Sparse block is missing from the static layout plan");
}
matrix_ref sparse_matrix::view(size_t r_st, size_t c_st, size_t r, size_t c,
                               sparsity sp) {
  static matrix empty;
  static vector empty_vec;
  static row_vector empty_rvec;
  if (r == 0 || c == 0)
    if (c == 1 || sp == sparsity::eye || sp == sparsity::diag)
      return empty_vec;
    else if (r == 1)
      return empty_rvec;
    else
      return empty;
  const sparse_block_spec requested{r_st, c_st, r, c, sp};
  for (const auto &binding : planned_) {
    if (binding.block != requested) continue;
    if (binding.storage_pattern == sparsity::dense)
      return dense_panels_[binding.panel].data_.block(
          binding.local_row, binding.local_col, r, c);
    if (binding.storage_pattern == sparsity::diag) {
      const auto &segment = diagonal_segments_[binding.panel];
      return diag_panels_[segment.storage_panel].data_.segment(
          segment.storage_offset + binding.local_row, r);
    }
    return eye_panels_[binding.panel].data_.segment(binding.local_row, r);
  }
  throw std::logic_error("Sparse block is missing from the static layout plan");
}
matrix sparse_matrix::dense() const {
  matrix out(rows_, cols_);
  out.setZero();
  linear_backend::run_dense_write(*this, out.data(), out.rows(), 1., false);
  return out;
}
} // namespace moto
