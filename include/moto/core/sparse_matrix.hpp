#ifndef MOTO_CORE_SPARSE_MATRIX_HPP
#define MOTO_CORE_SPARSE_MATRIX_HPP

#include <moto/core/sparse_panel.hpp>
#include <moto/core/codegen_context.hpp>

#include <span>

namespace moto {
namespace linear_backend {
struct matrix_cache;
}
struct sparse_matrix {
private:
  mutable std::filesystem::path codegen_dir_;
  /// Internal callers already own a resolved directory; never consult the
  /// filesystem when binding from a context or copying runtime matrices.
  void bind_resolved_codegen_directory(const std::filesystem::path &directory) {
    if (codegen_dir_ == directory) return;
    if (!codegen_dir_.empty())
      throw std::invalid_argument("sparse matrix uses codegen directory '" +
          codegen_dir_.string() + "', but requested '" + directory.string() +
          "'; create a fresh model for a different directory");
    codegen_dir_ = directory;
  }
public:
  /// Runtime matrices keep their directory even when kernels compile lazily.
  void bind_codegen_directory(const std::filesystem::path &directory) {
    bind_resolved_codegen_directory(std::filesystem::weakly_canonical(
        std::filesystem::absolute(directory)));
  }
  void bind_codegen(const codegen_context_ptr &context) {
    bind_resolved_codegen_directory(context->linear_dir());
  }
  const std::filesystem::path &linear_codegen_dir() const {
    if (codegen_dir_.empty())
      codegen_dir_ = std::filesystem::weakly_canonical(
          std::filesystem::absolute("gen/linear_backend"));
    return codegen_dir_;
  }
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
  sparse_matrix(const sparse_matrix &other);
  sparse_matrix(sparse_matrix &&) noexcept = default;
  bool is_empty() const {
    return dense_panels_.empty() && diag_panels_.empty() && eye_panels_.empty();
  }
  sparse_matrix &operator=(const sparse_matrix &other);
  sparse_matrix &operator=(sparse_matrix &&) noexcept = default;
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
  /// Maximum absolute row sum, or maximum absolute column sum when
  /// `transpose` is true.  Canonical panel layouts are evaluated without
  /// materializing the full matrix.
  scalar_t induced_inf_norm(bool transpose = false) const;
};

} // namespace moto
#endif
