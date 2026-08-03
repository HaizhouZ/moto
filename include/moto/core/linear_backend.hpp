#ifndef MOTO_LINEAR_JIT_BACKEND_HPP
#define MOTO_LINEAR_JIT_BACKEND_HPP

#include <moto/core/sparse_matrix.hpp>

#include <filesystem>
#include <span>

namespace moto::linear_backend {

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
};

struct matrix_layout {
  size_t rows = 0;
  size_t cols = 0;
  std::vector<panel_layout> panels;
};

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

std::string emit_product_source(const product_spec &spec);
product_kernel
compile_product(product_spec spec,
                const std::filesystem::path &cache_dir = "gen/linear_backend");
batch_product_kernel compile_batch_product(
    batch_product_spec spec,
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
void prepare_sparse_product(const ::moto::sparse_matrix &sparse,
                            const ::moto::sparse_matrix &other, product_op op,
                            scalar_t sign, size_t out_rows, size_t out_cols);
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
