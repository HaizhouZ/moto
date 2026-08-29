#ifndef MOTO_LINEAR_JIT_BACKEND_HPP
#define MOTO_LINEAR_JIT_BACKEND_HPP

#include <moto/core/sparse_matrix.hpp>

#include <filesystem>
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

matrix_layout describe(const casadi::Sparsity &sparsity);

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

private:
  size_t pointers_ = 0;
  function_type function_ = nullptr;
};

product_kernel
compile_product(product_spec spec,
                const std::filesystem::path &cache_dir = "gen/linear_backend");
batch_product_kernel compile_batch_product(
    batch_product_spec spec,
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

matrix_layout describe(const ::moto::sparse_matrix &sparse);
std::vector<scalar_t *> panel_pointers(const ::moto::sparse_matrix &sparse);
void run_product(const ::moto::sparse_matrix &sparse, product_op op,
                 scalar_t sign, const scalar_t *other, size_t other_rows,
                 size_t other_cols, scalar_t *out, size_t out_rows,
                 size_t out_cols);
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
void prepare_sparse_product(const ::moto::sparse_matrix &sparse,
                            const ::moto::sparse_matrix &other, product_op op,
                            scalar_t sign, size_t out_rows, size_t out_cols);
void run_dense_write(const ::moto::sparse_matrix &sparse, scalar_t *out,
                     size_t out_rows, scalar_t alpha, bool overwrite);
void prepare_dense_write(const ::moto::sparse_matrix &sparse, size_t out_rows,
                         scalar_t alpha, bool overwrite);

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

struct condensation_spec {
  size_t rows = 0;
  std::vector<size_t> jac_cols;
  std::vector<panel_layout> jacobians;
  std::vector<std::pair<size_t, size_t>> hessian_pairs;
  std::vector<scalar_t> residual_signs;

  void validate() const;
  size_t argument_count() const;
  size_t pointer_count() const;
};

batch_product_kernel compile_batch_jacobian_product(
    std::vector<std::pair<size_t, std::vector<panel_layout>>> products,
    const std::filesystem::path &cache_dir = "gen/linear_backend");

struct batch_condensation_spec {
  std::vector<condensation_spec> constraints;
};

using batch_condensation_kernel = batch_product_kernel;

batch_condensation_kernel compile_batch_condensation(
    batch_condensation_spec spec,
    const std::filesystem::path &cache_dir = "gen/linear_backend");

} // namespace moto::linear_backend

#endif
