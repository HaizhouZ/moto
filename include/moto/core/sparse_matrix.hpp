#ifndef MOTO_CORE_SPARSE_MATRIX_HPP
#define MOTO_CORE_SPARSE_MATRIX_HPP

#include <moto/core/sparse_panel.hpp>

#include <span>

namespace moto {
namespace linear_backend {
struct matrix_cache;
}
struct sparse_matrix {
  mutable std::shared_ptr<linear_backend::matrix_cache> jit_cache_;
  size_t rows_ = 0;
  size_t cols_ = 0;
  size_t rows() const { return rows_; }
  size_t cols() const { return cols_; }
  std::vector<sparse_panel<sparsity::dense>> dense_panels_;
  std::vector<sparse_panel<sparsity::diag>> diag_panels_;
  std::vector<sparse_panel<sparsity::eye>> eye_panels_;
  struct diagonal_segment {
    size_t row = 0, col = 0, rows = 0, cols = 0;
    size_t storage_panel = 0, storage_offset = 0;
  };
  std::vector<diagonal_segment> diagonal_segments_;
  bool packed_diagonal_storage_ = false;
  bool dynamic_eye_ = false;
  struct planned_binding {
    sparse_block_spec block;
    sparsity storage_pattern = sparsity::unknown;
    size_t panel = 0, local_row = 0, local_col = 0;
    bool used = false;
  };
  std::vector<planned_binding> planned_;
  sparse_matrix() = default;
  bool is_empty() const {
    return dense_panels_.empty() && diag_panels_.empty() && eye_panels_.empty();
  }
  sparse_matrix &operator=(const sparse_matrix &other);
  void setZero();
  void resize(size_t rows, size_t cols);
  bool valid() const;
  bool set_dynamic_eye(bool enabled);
  matrix_ref insert(size_t r_st, size_t c_st, size_t r, size_t c, sparsity sp);
  matrix_ref bind(size_t r_st, size_t c_st, size_t r, size_t c, sparsity sp);
  /// Return a repeatable view of a statically planned binding. Unlike bind(),
  /// this does not consume the binding and is intended for runtime callbacks.
  matrix_ref view(size_t r_st, size_t c_st, size_t r, size_t c, sparsity sp);
  void plan(const sparse_layout_plan &layout);
  void plan(std::span<const sparse_block_spec> blocks,
            sparse_plan_mode mode = sparse_plan_mode::distinct);
  template <sparsity Sp>
  matrix_ref insert(size_t r_st, size_t c_st, size_t dim) {
    return insert(r_st, c_st, dim, dim, Sp);
  }
  matrix dense() const;
  template <typename rhs_type> sparse_matrix &operator=(const rhs_type &rhs) {
    for (auto &panel : dense_panels_) {
      if (panel.rows_ == rhs.rows())
        panel.data_.noalias() = rhs.middleCols(panel.col_st_, panel.cols_);
    }
    return *this;
  }
};

} // namespace moto
#endif
