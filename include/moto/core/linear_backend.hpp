#ifndef MOTO_LINEAR_JIT_BACKEND_HPP
#define MOTO_LINEAR_JIT_BACKEND_HPP

#include <moto/core/sparse_matrix.hpp>

#include <filesystem>
#include <limits>
#include <span>
#include <string_view>

namespace casadi { class MX; class Sparsity; }

namespace moto::linear_backend {

namespace detail {
struct casadi_mx_graph_plan;
struct casadi_mx_graph_instance;
}

enum class product_op {
  times,
  transpose_times,
  right_times,
  right_transpose_times
};

struct panel_layout {
  sparsity pattern;
  size_t row_offset;
  size_t col_offset;
  size_t rows;
  size_t cols;
  /// Logical panel may be a zero-copy transpose of its physical storage.
  /// storage_offset is measured in scalars from the bound panel pointer.
  bool transposed = false;
  size_t storage_offset = 0;
  /// Physical column stride for a view. Zero selects its natural stride.
  size_t storage_rows = 0;

  bool operator==(const panel_layout &) const = default;
};

struct matrix_layout {
  size_t rows = 0;
  size_t cols = 0;
  std::vector<panel_layout> panels;

  bool operator==(const matrix_layout &) const = default;
};

/// Canonical algebraic sparsity used by graph optimizers. Unlike
/// matrix_layout this contains no storage, alignment, or panelization choices,
/// so equivalent expressions can share one e-class even when their extracted
/// layouts differ.
struct sparse_pattern {
  size_t rows = 0;
  size_t cols = 0;
  std::vector<size_t> colind;
  std::vector<size_t> row;

  size_t nnz() const { return row.size(); }
  void validate() const;
  bool operator==(const sparse_pattern &) const = default;
};

struct sparse_pattern_hash {
  size_t operator()(const sparse_pattern &pattern) const noexcept;
};

sparse_pattern analyze_pattern(const matrix_layout &layout,
                               bool transpose = false);
sparse_pattern transpose_pattern(const sparse_pattern &pattern);
sparse_pattern analyze_spmm_pattern(const sparse_pattern &lhs,
                                    const sparse_pattern &rhs);
matrix_layout panelize_pattern(const sparse_pattern &pattern);

/// A logical operand of an SpMM expression. transpose is algebraic; panel
/// transposed flags continue to describe physical storage views.
struct spmm_operand {
  matrix_layout layout;
  bool transpose = false;
  /// Optional exact semantic pattern. Intermediates keep this when their
  /// executable layout is conservatively enlarged to cover whole kernels.
  sparse_pattern pattern;

  size_t rows() const { return transpose ? layout.cols : layout.rows; }
  size_t cols() const { return transpose ? layout.rows : layout.cols; }
  bool operator==(const spmm_operand &) const = default;
};

struct spmm_panel_product {
  size_t lhs_panel = 0;
  size_t rhs_panel = 0;
  size_t reduction_begin = 0;
  size_t reduction_size = 0;
  sparse_block_spec output;
  size_t scalar_products = 0;

  bool operator==(const spmm_panel_product &) const = default;
};

/// Pure SpMM analysis suitable for e-class analysis and extraction. It owns no
/// data and compiles nothing.
struct spmm_analysis {
  spmm_operand lhs;
  spmm_operand rhs;
  sparse_pattern output_pattern;
  matrix_layout output_layout;
  std::vector<spmm_panel_product> products;

  size_t scalar_products() const;
  bool operator==(const spmm_analysis &) const = default;
};

spmm_analysis analyze_spmm(spmm_operand lhs, spmm_operand rhs);

struct product_spec {
  matrix_layout sparse;
  product_op op = product_op::times;
  size_t other_rows = 0;
  size_t other_cols = 0;
  size_t out_rows = 0;
  size_t out_cols = 0;
  scalar_t sign = 1.;

  void validate() const;
};

class product_kernel {
public:
  using function_type = void (*)(scalar_t *const *, const scalar_t *,
                                 scalar_t *);
  product_kernel() = default;
  product_kernel(product_spec spec, function_type function)
      : spec_(std::move(spec)), function_(function) {}
  void operator()(std::span<scalar_t *> pointers) const;
  void operator()(std::span<scalar_t *> panels, const scalar_t *other,
                  scalar_t *output) const;
  size_t pointer_count() const { return spec_.sparse.panels.size() + 2; }
  size_t other_slot() const { return spec_.sparse.panels.size(); }
  size_t output_slot() const { return spec_.sparse.panels.size() + 1; }

private:
  product_spec spec_;
  function_type function_ = nullptr;
};

struct batch_product_spec {
  std::vector<product_spec> products;
  void validate() const;
};

class batch_product_kernel {
public:
  using function_type = void (*)(scalar_t *const *);
  batch_product_kernel() = default;
  batch_product_kernel(size_t pointers, function_type function)
      : pointers_(pointers), function_(function) {}
  void operator()(std::span<scalar_t *> pointers) const;
  size_t pointer_count() const { return pointers_; }
  function_type function() const { return function_; }

private:
  size_t pointers_ = 0;
  function_type function_ = nullptr;
};

std::string emit_product_source(const product_spec &spec);
product_kernel
compile_product(product_spec spec,
                const std::filesystem::path &cache_dir = "gen/linear_backend");
batch_product_kernel compile_batch_product(
    batch_product_spec spec,
    const std::filesystem::path &cache_dir = "gen/linear_backend");

/// Compile one panel-to-panel SpMM operation. Pointer order is all effective
/// lhs panels, all effective rhs panels, then all output panels.
batch_product_kernel compile_sparse_product(
    matrix_layout lhs, matrix_layout rhs, product_op op, scalar_t sign,
    matrix_layout output,
    const std::filesystem::path &cache_dir = "gen/linear_backend");
/// Compile a product selected by a graph extractor. The output layout is the
/// canonical panelization carried by analysis.
batch_product_kernel compile_sparse_product(
    const spmm_analysis &analysis, scalar_t sign = 1.,
    const std::filesystem::path &cache_dir = "gen/linear_backend");
/// Compile a sparse product against an existing graph-wide pointer table.
/// Slot arrays map lhs/rhs/output panels directly into that table, avoiding a
/// per-call gather of panel pointers.
batch_product_kernel compile_indexed_sparse_product(
    matrix_layout lhs, matrix_layout rhs, product_op op, scalar_t sign,
    matrix_layout output, size_t pointer_count,
    std::span<const size_t> lhs_slots, std::span<const size_t> rhs_slots,
    std::span<const size_t> output_slots,
    const std::filesystem::path &cache_dir = "gen/linear_backend");
batch_product_kernel compile_indexed_sparse_product(
    const spmm_analysis &analysis, scalar_t sign, size_t pointer_count,
    std::span<const size_t> lhs_slots, std::span<const size_t> rhs_slots,
    std::span<const size_t> output_slots,
    const std::filesystem::path &cache_dir = "gen/linear_backend");

/// Scalar transfers surrounding panel products and factorizations.  The MX
/// translator only describes addresses and algebra; this backend compiles the
/// description into one straight-line kernel for each dependency-preserving
/// region of the graph.
enum class panel_program_op { copy, fill, add, sub, mul, div };

struct panel_program_operand {
  static constexpr size_t invalid = std::numeric_limits<size_t>::max();
  size_t pointer = invalid;
  size_t offset = 0;
  ptrdiff_t stride = 0;
};

struct panel_program_instruction {
  panel_program_op op = panel_program_op::copy;
  panel_program_operand destination, lhs, rhs;
  size_t count = 1;
  scalar_t scalar = 1.;
};

struct panel_program_spec {
  size_t pointers = 0;
  std::vector<panel_program_instruction> instructions;
  void validate() const;
};

panel_program_spec coalesce_panel_program_spec(panel_program_spec spec);

/// Compile output initialization and a panel product into one generated
/// kernel.  The initialization is the lazy destination expression (typically
/// C or -C in C +/- A*B); product contributions accumulate immediately into
/// that storage without materializing A*B.
batch_product_kernel compile_indexed_sparse_product_lazy(
    matrix_layout lhs, matrix_layout rhs, product_op op, scalar_t sign,
    matrix_layout output, size_t pointer_count,
    std::span<const size_t> lhs_slots, std::span<const size_t> rhs_slots,
    std::span<const size_t> output_slots, panel_program_spec initialization,
    const std::filesystem::path &cache_dir = "gen/linear_backend");

class panel_program_kernel {
public:
  using function_type = void (*)(scalar_t *const *);
  panel_program_kernel() = default;
  panel_program_kernel(size_t pointers, function_type function)
      : pointers_(pointers), function_(function) {}
  void operator()(std::span<scalar_t *> pointers) const;
  size_t pointer_count() const { return pointers_; }
  explicit operator bool() const { return function_ != nullptr; }

private:
  size_t pointers_ = 0;
  function_type function_ = nullptr;
};

panel_program_kernel compile_panel_program(
    panel_program_spec spec,
    const std::filesystem::path &cache_dir = "gen/linear_backend");

/// Lowered MX elimination schedule using the backend's panel kernels.  MX is
/// compile-time DAG/sparsity metadata only.  Existing Jacobian panels are
/// caller-owned inputs; the schedule must not regenerate or repack them.
class graph_kernel {
public:
  graph_kernel();
  graph_kernel(graph_kernel &&) noexcept;
  graph_kernel &operator=(graph_kernel &&) noexcept;
  graph_kernel(const graph_kernel &) = delete;
  graph_kernel &operator=(const graph_kernel &) = delete;
  ~graph_kernel();
  void operator()(std::span<scalar_t *> pointers) const;
  void operator()(size_t entry, std::span<scalar_t *> pointers) const;
  size_t input_count() const;
  size_t output_count() const;
  size_t entry_count() const;
  size_t pointer_count() const { return inputs_ + outputs_; }
  graph_kernel instantiate(
      std::vector<::moto::sparse_matrix> *workspace = nullptr) const;
  explicit operator bool() const { return instance_ != nullptr; }

private:
  friend graph_kernel compile_graph(
      const std::vector<casadi::MX> &, const std::vector<casadi::MX> &,
      std::vector<::moto::sparse_matrix> *, const std::filesystem::path &);
  friend graph_kernel compile_graph(
      const std::vector<casadi::MX> &,
      const std::vector<std::vector<casadi::MX>> &,
      std::vector<::moto::sparse_matrix> *, const std::filesystem::path &);
  friend graph_kernel compile_graph(
      const std::vector<casadi::MX> &,
      const std::vector<std::vector<casadi::MX>> &,
      std::span<const matrix_layout>,
      std::vector<::moto::sparse_matrix> *, const std::filesystem::path &);
  friend graph_kernel compile_graph(
      std::string_view, const std::vector<casadi::MX> &,
      const std::vector<std::vector<casadi::MX>> &,
      std::span<const matrix_layout>,
      std::vector<::moto::sparse_matrix> *, const std::filesystem::path &,
      std::span<const casadi::MX>);
  explicit graph_kernel(
      std::unique_ptr<detail::casadi_mx_graph_instance> instance);
  size_t inputs_ = 0, outputs_ = 0, entries_ = 0;
  std::unique_ptr<detail::casadi_mx_graph_instance> instance_;
};

graph_kernel compile_graph(
    const std::vector<casadi::MX> &inputs,
    const std::vector<casadi::MX> &outputs,
    std::vector<::moto::sparse_matrix> *workspace,
    const std::filesystem::path &cache_dir = "gen/linear_backend");
graph_kernel compile_graph(
    const std::vector<casadi::MX> &inputs,
    const std::vector<std::vector<casadi::MX>> &output_entries,
    std::vector<::moto::sparse_matrix> *workspace,
    const std::filesystem::path &cache_dir = "gen/linear_backend");
graph_kernel compile_graph(
    const std::vector<casadi::MX> &inputs,
    const std::vector<std::vector<casadi::MX>> &output_entries,
    std::span<const matrix_layout> input_layouts,
    std::vector<::moto::sparse_matrix> *workspace,
    const std::filesystem::path &cache_dir = "gen/linear_backend");
graph_kernel compile_graph(
    std::string_view artifact_identity,
    const std::vector<casadi::MX> &inputs,
    const std::vector<std::vector<casadi::MX>> &output_entries,
    std::span<const matrix_layout> input_layouts,
    std::vector<::moto::sparse_matrix> *workspace,
    const std::filesystem::path &cache_dir = "gen/linear_backend",
    std::span<const casadi::MX> spd_factors = {});
inline graph_kernel compile_graph(
    const std::vector<casadi::MX> &inputs,
    const std::vector<casadi::MX> &outputs,
    const std::filesystem::path &cache_dir = "gen/linear_backend") {
  return compile_graph(inputs, outputs, nullptr, cache_dir);
}

struct ccs_layout {
  size_t rows = 0, cols = 0;
  std::vector<size_t> colind, row;
  std::vector<size_t> row_permutation, col_permutation;
  std::vector<size_t> row_blocks, col_blocks;
  size_t nnz() const { return row.size(); }
};

class spgemm_kernel {
public:
  using function_type = void (*)(scalar_t *const *);
  spgemm_kernel() = default;
  spgemm_kernel(size_t lhs_nnz, size_t rhs_nnz, ccs_layout output,
                function_type function)
      : lhs_nnz_(lhs_nnz), rhs_nnz_(rhs_nnz), output_(std::move(output)),
        function_(function) {}
  void operator()(const scalar_t *lhs, const scalar_t *rhs,
                  scalar_t *output) const;
  const ccs_layout &output_layout() const { return output_; }

private:
  size_t lhs_nnz_ = 0, rhs_nnz_ = 0;
  ccs_layout output_;
  function_type function_ = nullptr;
};

ccs_layout analyze_spgemm(const casadi::Sparsity &lhs,
                          const casadi::Sparsity &rhs);
ccs_layout analyze_sparsity(const casadi::Sparsity &sparsity);
spgemm_kernel compile_spgemm(
    const casadi::Sparsity &lhs, const casadi::Sparsity &rhs,
    const std::filesystem::path &cache_dir = "gen/linear_backend");

matrix_layout describe(const ::moto::sparse_matrix &sparse);
std::vector<scalar_t *> panel_pointers(const ::moto::sparse_matrix &sparse);
void run_product(const ::moto::sparse_matrix &sparse, product_op op,
                 scalar_t sign, const scalar_t *other, size_t other_rows,
                 size_t other_cols, scalar_t *out, size_t out_rows,
                 size_t out_cols);
void prepare_product(const ::moto::sparse_matrix &sparse, product_op op,
                     scalar_t sign, size_t other_rows, size_t other_cols,
                     size_t out_rows, size_t out_cols);
struct product_request {
  const ::moto::sparse_matrix *sparse;
  product_op op;
  scalar_t sign;
  size_t other_rows;
  size_t other_cols;
  size_t out_rows;
  size_t out_cols;
};
void prepare_products(std::span<const product_request> requests);
void run_sparse_product(const ::moto::sparse_matrix &sparse,
                        const ::moto::sparse_matrix &other, product_op op,
                        scalar_t sign, scalar_t *out, size_t out_rows,
                        size_t out_cols);
/// Execute directly between planned sparse-matrix panel stores. This is the
/// graph-runtime path: no CSC repacking and no dense staging matrix.
void run_sparse_product(const ::moto::sparse_matrix &sparse,
                        const ::moto::sparse_matrix &other, product_op op,
                        scalar_t sign, ::moto::sparse_matrix &out);
void prepare_sparse_product(const ::moto::sparse_matrix &sparse,
                            const ::moto::sparse_matrix &other, product_op op,
                            scalar_t sign, size_t out_rows, size_t out_cols);
void prepare_sparse_product(const ::moto::sparse_matrix &sparse,
                            const ::moto::sparse_matrix &other, product_op op,
                            scalar_t sign, ::moto::sparse_matrix &out);
void run_dense_write(const ::moto::sparse_matrix &sparse, scalar_t *out,
                     size_t out_rows, scalar_t alpha, bool overwrite);
void prepare_dense_write(const ::moto::sparse_matrix &sparse, size_t out_rows,
                         scalar_t alpha, bool overwrite);
void run_weighted_gram(const ::moto::sparse_matrix &sparse,
                       const scalar_t *middle, scalar_t *out);
void prepare_weighted_gram(const ::moto::sparse_matrix &sparse);

enum class rowwise_op { scale, inf_norm, scaled_inf_norm };

class rowwise_kernel {
public:
  using function_type = void (*)(scalar_t *const *, const scalar_t *, scalar_t *);
  rowwise_kernel() = default;
  rowwise_kernel(size_t panels, function_type function)
      : panels_(panels), function_(function) {}
  void operator()(std::span<scalar_t *const> panels, const scalar_t *scale,
                  scalar_t *output) const;

private:
  size_t panels_ = 0;
  function_type function_ = nullptr;
};

struct rowwise_kernels {
  rowwise_kernel scale, inf_norm, scaled_inf_norm;
};

rowwise_kernels compile_rowwise(
    matrix_layout layout,
    const std::filesystem::path &cache_dir = "gen/linear_backend");

struct dense_write_config {
  scalar_t alpha = 1.;
  bool overwrite = false;
};

template <typename Rhs, typename Out>
void multiply(const sparse_matrix &sparse, const Rhs &rhs, Out &out,
              scalar_t alpha = 1.) {
  if constexpr (std::is_same_v<std::remove_cvref_t<Rhs>, sparse_matrix>)
    run_sparse_product(sparse, rhs, product_op::times, alpha, out.data(),
                       sparse.rows(), rhs.cols());
  else
    run_product(sparse, product_op::times, alpha, rhs.data(), rhs.rows(),
                rhs.cols(), out.data(), sparse.rows(), rhs.cols());
}

template <typename Rhs, typename Out>
void transpose_multiply(const sparse_matrix &sparse, const Rhs &rhs, Out &out,
                        scalar_t alpha = 1.) {
  if constexpr (std::is_same_v<std::remove_cvref_t<Rhs>, sparse_matrix>)
    run_sparse_product(sparse, rhs, product_op::transpose_times, alpha,
                       out.data(), sparse.cols(), rhs.cols());
  else
    run_product(sparse, product_op::transpose_times, alpha, rhs.data(),
                rhs.rows(), rhs.cols(), out.data(), sparse.cols(), rhs.cols());
}

template <typename Lhs, typename Out>
void right_multiply(const Lhs &lhs, const sparse_matrix &sparse, Out &out,
                    scalar_t alpha = 1.) {
  if constexpr (std::is_same_v<std::remove_cvref_t<Lhs>, sparse_matrix>)
    run_sparse_product(sparse, lhs, product_op::right_times, alpha, out.data(),
                       lhs.rows(), sparse.cols());
  else
    run_product(sparse, product_op::right_times, alpha, lhs.data(), lhs.rows(),
                lhs.cols(), out.data(), lhs.rows(), sparse.cols());
}

template <typename Lhs, typename Out>
void right_transpose_multiply(const Lhs &lhs, const sparse_matrix &sparse,
                              Out &out, scalar_t alpha = 1.) {
  if constexpr (std::is_same_v<std::remove_cvref_t<Lhs>, sparse_matrix>)
    run_sparse_product(sparse, lhs, product_op::right_transpose_times, alpha,
                       out.data(), lhs.cols(), sparse.cols());
  else
    run_product(sparse, product_op::right_transpose_times, alpha, lhs.data(),
                lhs.rows(), lhs.cols(), out.data(), lhs.cols(), sparse.cols());
}

template <typename Out>
void write_dense(const sparse_matrix &sparse, Out &&out,
                 dense_write_config config = {}) {
  run_dense_write(sparse, out.data(), out.rows(), config.alpha,
                  config.overwrite);
}

template <typename Middle, typename Out>
void weighted_gram(const sparse_matrix &sparse, const Middle &middle,
                   Out &out) {
  run_weighted_gram(sparse, middle.data(), out.data());
}

struct condensation_spec {
  size_t rows = 0;
  std::vector<size_t> jac_cols;
  std::vector<panel_layout> jacobians;
  std::vector<std::pair<size_t, size_t>> hessian_pairs;
  std::vector<scalar_t> residual_signs;

  void validate() const;
  size_t argument_count() const;
};

panel_layout condensation_hessian_layout(const panel_layout &lhs,
                                         const panel_layout &rhs);

batch_product_kernel compile_batch_jacobian_product(
    std::vector<std::pair<size_t, std::vector<panel_layout>>> products,
    const std::filesystem::path &cache_dir = "gen/linear_backend");

class condensation_kernel {
public:
  using function_type = void (*)(scalar_t *const *);

  condensation_kernel() = default;
  condensation_kernel(condensation_spec spec, function_type function,
                      std::string cache_key)
      : spec_(std::move(spec)), function_(function),
        cache_key_(std::move(cache_key)) {}

  explicit operator bool() const { return function_ != nullptr; }
  void operator()(std::span<scalar_t *> pointers) const;

  size_t pointer_count() const;
  size_t jacobian_slot(size_t arg) const;
  size_t residual_slot(size_t side) const;
  size_t weight_slot(size_t side) const;
  size_t gradient_slot(size_t arg) const;
  size_t hessian_slot(size_t lhs, size_t rhs) const;
  const std::string &cache_key() const { return cache_key_; }

private:
  condensation_spec spec_;
  function_type function_ = nullptr;
  std::string cache_key_;
};

struct batch_condensation_spec {
  std::vector<condensation_spec> constraints;
};

using batch_condensation_kernel = batch_product_kernel;

std::string emit_condensation_source(const condensation_spec &spec);
condensation_kernel compile_condensation(
    condensation_spec spec,
    const std::filesystem::path &cache_dir = "gen/linear_backend");
batch_condensation_kernel compile_batch_condensation(
    batch_condensation_spec spec,
    const std::filesystem::path &cache_dir = "gen/linear_backend");

} // namespace moto::linear_backend

#endif
