#include <moto/core/linear_backend.hpp>

#include <moto/core/external_function.hpp>
#include <moto/core/sparse_matrix.hpp>
#include <moto/utils/codegen.hpp>

#include <algorithm>
#include <cstdlib>
#include <dlfcn.h>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <unordered_map>

namespace moto::linear_backend {
namespace {

constexpr std::string_view symbol_name = "moto_linear_jit_kernel";
std::mutex compile_mutex;
std::unordered_map<std::string, void *> loaded_kernels;
std::unordered_map<std::string, std::shared_ptr<std::mutex>> compile_locks;
std::mutex batch_mutex;
std::unordered_map<std::string, batch_product_kernel> batch_kernels;
std::unordered_map<std::string, batch_condensation_kernel> condensation_kernels;
std::unordered_map<std::string, batch_product_kernel> jacobian_product_kernels;

std::string shell_quote(const std::filesystem::path &path) {
  std::string out = "'";
  for (const char c : path.string())
    out += c == '\'' ? "'\\''" : std::string(1, c);
  return out + "'";
}

size_t pair_index(size_t n, size_t lhs, size_t rhs) {
  if (lhs > rhs)
    std::swap(lhs, rhs);
  return lhs * n - lhs * (lhs - 1) / 2 + rhs - lhs;
}

std::vector<std::vector<size_t>>
fused_panel_groups(const matrix_layout &layout) {
  using key_t = std::tuple<sparsity, size_t, size_t, size_t, size_t>;
  std::map<key_t, std::vector<size_t>> grouped;
  for (size_t i = 0; i < layout.panels.size(); ++i) {
    const auto &p = layout.panels[i];
    grouped[{p.pattern, p.row_offset, p.col_offset, p.rows, p.cols}]
        .push_back(i);
  }
  std::vector<std::vector<size_t>> result;
  for (auto &[key, panels] : grouped)
    if (panels.size() > 1)
      result.push_back(std::move(panels));
  return result;
}

std::vector<std::pair<size_t, size_t>>
condensation_pairs(const condensation_spec &spec) {
  if (!spec.hessian_pairs.empty())
    return spec.hessian_pairs;
  std::vector<std::pair<size_t, size_t>> pairs;
  for (size_t i = 0; i < spec.argument_count(); ++i)
    for (size_t j = i; j < spec.argument_count(); ++j)
      pairs.emplace_back(i, j);
  return pairs;
}

void *compile_source(const std::string &source,
                     const std::filesystem::path &cache_dir) {
  const std::string key = utils::compute_md5_from_bytes(source);
  const auto cpp = cache_dir / (key + ".cpp");
  const auto lib = cache_dir / ("lib" + key + ".so");
  const auto tmp = cache_dir / ("lib" + key + ".so.tmp");
  std::shared_ptr<std::mutex> key_mutex;
  {
    std::lock_guard lock(compile_mutex);
    if (auto it = loaded_kernels.find(key); it != loaded_kernels.end())
      return it->second;
    auto &slot = compile_locks[key];
    if (!slot)
      slot = std::make_shared<std::mutex>();
    key_mutex = slot;
  }
  std::lock_guard key_lock(*key_mutex);
  {
    std::lock_guard lock(compile_mutex);
    if (auto it = loaded_kernels.find(key); it != loaded_kernels.end())
      return it->second;
  }
  {
    std::filesystem::create_directories(cache_dir);
    if (!std::filesystem::exists(lib)) {
      std::ofstream out(cpp);
      if (!out)
        throw std::runtime_error("failed to create sparse JIT source");
      out << source;
      out.close();
      const std::string command =
          "g++ -shared -fPIC -std=c++20 -O3 -DNDEBUG -march=native "
          "-fopenmp-simd -ffp-contract=fast -I/usr/include/eigen3 -o " +
          shell_quote(tmp) + " " + shell_quote(cpp) + [&] {
            Dl_info info{};
            return dladdr(reinterpret_cast<void *>(&compile_source), &info) &&
                           info.dli_fname
                       ? " " + shell_quote(info.dli_fname)
                       : std::string{};
          }();
      if (const int status = std::system(command.c_str()); status != 0) {
        std::error_code ec;
        std::filesystem::remove(tmp, ec);
        throw std::runtime_error("failed to compile sparse JIT kernel");
      }
      std::filesystem::rename(tmp, lib);
    }
  }
  void *function = load_from_shared(lib.string(), std::string(symbol_name));
  {
    std::lock_guard lock(compile_mutex);
    loaded_kernels.emplace(key, function);
  }
  return function;
}

} // namespace

extern "C" __attribute__((visibility("default"))) void
moto_linear_dense_times(const double *a, size_t ar, size_t ac, const double *b,
                        size_t br, size_t bc, double *o, size_t orows,
                        size_t ocols, size_t ro, size_t co, double alpha) {
  Eigen::Map<const matrix, Eigen::Aligned> A(a, ar, ac);
  Eigen::Map<const matrix> B(b, br, bc);
  Eigen::Map<matrix> O(o, orows, ocols);
  O.block(ro, 0, ar, bc).noalias() += alpha * A * B.block(co, 0, ac, bc);
}

extern "C" __attribute__((visibility("default"))) void
moto_linear_dense_transpose_times(const double *a, size_t ar, size_t ac,
                                  const double *b, size_t br, size_t bc,
                                  double *o, size_t orows, size_t ocols,
                                  size_t ro, size_t co, double alpha) {
  Eigen::Map<const matrix, Eigen::Aligned> A(a, ar, ac);
  Eigen::Map<const matrix> B(b, br, bc);
  Eigen::Map<matrix> O(o, orows, ocols);
  O.block(co, 0, ac, bc).noalias() +=
      alpha * A.transpose() * B.block(ro, 0, ar, bc);
}

extern "C" __attribute__((visibility("default"))) void
moto_linear_dense_right_times(const double *a, size_t ar, size_t ac,
                              const double *b, size_t br, size_t bc, double *o,
                              size_t orows, size_t ocols, size_t ro, size_t co,
                              double alpha) {
  Eigen::Map<const matrix, Eigen::Aligned> A(a, ar, ac);
  Eigen::Map<const matrix> B(b, br, bc);
  Eigen::Map<matrix> O(o, orows, ocols);
  O.block(0, co, br, ac).noalias() += alpha * B.block(0, ro, br, ar) * A;
}

extern "C" __attribute__((visibility("default"))) void
moto_linear_dense_right_transpose_times(const double *a, size_t ar, size_t ac,
                                        const double *b, size_t br, size_t bc,
                                        double *o, size_t orows, size_t ocols,
                                        size_t ro, size_t co, double alpha) {
  Eigen::Map<const matrix, Eigen::Aligned> A(a, ar, ac);
  Eigen::Map<const matrix> B(b, br, bc);
  Eigen::Map<matrix> O(o, orows, ocols);
  O.block(0, co, bc, ac).noalias() +=
      alpha * B.block(ro, 0, ar, bc).transpose() * A;
}

template <product_op Op, bool Eye>
void structured_product(const double *a, size_t n, const double *b, size_t br,
                        size_t bc, double *o, size_t orows, size_t ocols,
                        size_t ro, size_t co, double alpha) {
  Eigen::Map<const matrix> B(b, br, bc);
  Eigen::Map<matrix> O(o, orows, ocols);
  if constexpr (Op == product_op::times) {
    if constexpr (Eye)
      O.middleRows(ro, n) += alpha * B.middleRows(co, n);
    else
      O.middleRows(ro, n).noalias() +=
          alpha * Eigen::Map<const vector, Eigen::Aligned>(a, n).asDiagonal() *
          B.middleRows(co, n);
  } else if constexpr (Op == product_op::transpose_times) {
    if constexpr (Eye)
      O.middleRows(co, n) += alpha * B.middleRows(ro, n);
    else
      O.middleRows(co, n).noalias() +=
          alpha * Eigen::Map<const vector, Eigen::Aligned>(a, n).asDiagonal() *
          B.middleRows(ro, n);
  } else if constexpr (Op == product_op::right_times) {
    if constexpr (Eye)
      O.middleCols(co, n) += alpha * B.middleCols(ro, n);
    else
      O.middleCols(co, n).noalias() +=
          alpha * B.middleCols(ro, n) *
          Eigen::Map<const vector, Eigen::Aligned>(a, n).asDiagonal();
  } else {
    if constexpr (Eye)
      O.middleCols(co, n) += alpha * B.middleRows(ro, n).transpose();
    else
      O.middleCols(co, n).noalias() +=
          alpha * B.middleRows(ro, n).transpose() *
          Eigen::Map<const vector, Eigen::Aligned>(a, n).asDiagonal();
  }
}

template <product_op Op, bool Dense>
void fused_pair_product(const double *a, const double *a1, size_t ar,
                        size_t ac, const double *b, size_t br, size_t bc,
                        double *o, size_t orows, size_t ocols, size_t ro,
                        size_t co, double alpha) {
  Eigen::Map<const matrix> B(b, br, bc);
  Eigen::Map<matrix> O(o, orows, ocols);
  if constexpr (Dense) {
    const Eigen::Map<const matrix, Eigen::Aligned> A(a, ar, ac), A1(a1, ar, ac);
    if constexpr (Op == product_op::times)
      O.block(ro, 0, ar, bc).noalias() +=
          alpha * (A + A1) * B.block(co, 0, ac, bc);
    else if constexpr (Op == product_op::transpose_times)
      O.block(co, 0, ac, bc).noalias() +=
          alpha * (A + A1).transpose() * B.block(ro, 0, ar, bc);
    else if constexpr (Op == product_op::right_times)
      O.block(0, co, br, ac).noalias() +=
          alpha * B.block(0, ro, br, ar) * (A + A1);
    else
      O.block(0, co, bc, ac).noalias() +=
          alpha * B.block(ro, 0, ar, bc).transpose() * (A + A1);
  } else {
    const auto d = Eigen::Map<const vector, Eigen::Aligned>(a, ar).array() +
                   Eigen::Map<const vector, Eigen::Aligned>(a1, ar).array();
    if constexpr (Op == product_op::times)
      O.middleRows(ro, ar).array() +=
          alpha * (B.middleRows(co, ar).array().colwise() * d);
    else if constexpr (Op == product_op::transpose_times)
      O.middleRows(co, ar).array() +=
          alpha * (B.middleRows(ro, ar).array().colwise() * d);
    else if constexpr (Op == product_op::right_times)
      O.middleCols(co, ar).array() +=
          alpha * (B.middleCols(ro, ar).array().rowwise() * d.transpose());
    else
      O.middleCols(co, ar).array() +=
          alpha * (B.middleRows(ro, ar).transpose().array().rowwise() *
                   d.transpose());
  }
}

#define MOTO_STRUCTURED_PRODUCT_WRAPPER(name, op, eye)                         \
  extern "C" __attribute__((visibility("default"))) void name(                 \
      const double *a, size_t ar, size_t, const double *b, size_t br,          \
      size_t bc, double *o, size_t orows, size_t ocols, size_t ro, size_t co,  \
      double alpha) {                                                          \
    structured_product<product_op::op, eye>(a, ar, b, br, bc, o, orows, ocols, \
                                            ro, co, alpha);                    \
  }
MOTO_STRUCTURED_PRODUCT_WRAPPER(moto_linear_diag_times, times, false)
MOTO_STRUCTURED_PRODUCT_WRAPPER(moto_linear_eye_times, times, true)
MOTO_STRUCTURED_PRODUCT_WRAPPER(moto_linear_diag_transpose_times,
                                transpose_times, false)
MOTO_STRUCTURED_PRODUCT_WRAPPER(moto_linear_eye_transpose_times,
                                transpose_times, true)
MOTO_STRUCTURED_PRODUCT_WRAPPER(moto_linear_diag_right_times, right_times,
                                false)
MOTO_STRUCTURED_PRODUCT_WRAPPER(moto_linear_eye_right_times, right_times, true)
MOTO_STRUCTURED_PRODUCT_WRAPPER(moto_linear_diag_right_transpose_times,
                                right_transpose_times, false)
MOTO_STRUCTURED_PRODUCT_WRAPPER(moto_linear_eye_right_transpose_times,
                                right_transpose_times, true)
#undef MOTO_STRUCTURED_PRODUCT_WRAPPER

#define MOTO_FUSED_PAIR_WRAPPER(pattern, dense, op)                            \
  extern "C" __attribute__((visibility("default"))) void                     \
      moto_linear_fused_##pattern##_##op(                                     \
          const double *a, const double *a1, size_t ar, size_t ac,             \
          const double *b, size_t br, size_t bc, double *o, size_t orows,      \
          size_t ocols, size_t ro, size_t co, double alpha) {                  \
    fused_pair_product<product_op::op, dense>(                                 \
        a, a1, ar, ac, b, br, bc, o, orows, ocols, ro, co, alpha);             \
  }
#define MOTO_FUSED_PAIR_OPS(pattern, dense)                                    \
  MOTO_FUSED_PAIR_WRAPPER(pattern, dense, times)                               \
  MOTO_FUSED_PAIR_WRAPPER(pattern, dense, transpose_times)                     \
  MOTO_FUSED_PAIR_WRAPPER(pattern, dense, right_times)                         \
  MOTO_FUSED_PAIR_WRAPPER(pattern, dense, right_transpose_times)
MOTO_FUSED_PAIR_OPS(dense, true)
MOTO_FUSED_PAIR_OPS(diag, false)
#undef MOTO_FUSED_PAIR_OPS
#undef MOTO_FUSED_PAIR_WRAPPER

template <sparsity Pattern, bool Overwrite>
void dump_panel(const double *a, size_t rows, size_t cols, double *out,
                size_t out_rows, size_t row_offset, size_t col_offset,
                double alpha) {
  Eigen::Map<matrix> O(out, out_rows, col_offset + cols);
  if constexpr (Pattern == sparsity::dense) {
    const Eigen::Map<const matrix, Eigen::Aligned> A(a, rows, cols);
    if constexpr (Overwrite)
      O.block(row_offset, col_offset, rows, cols) = alpha * A;
    else
      O.block(row_offset, col_offset, rows, cols) += alpha * A;
  } else {
    auto diagonal = O.block(row_offset, col_offset, rows, rows).diagonal();
    if constexpr (Pattern == sparsity::diag) {
      const Eigen::Map<const vector, Eigen::Aligned> A(a, rows);
      if constexpr (Overwrite)
        diagonal = alpha * A;
      else
        diagonal += alpha * A;
    } else if constexpr (Overwrite) {
      diagonal.setConstant(alpha);
    } else {
      diagonal.array() += alpha;
    }
  }
}

template <sparsity Pattern, bool Overwrite>
void dump_pair(const double *a, const double *a1, size_t rows, size_t cols,
               double *out, size_t out_rows, size_t row, size_t col,
               double alpha) {
  Eigen::Map<matrix> O(out, out_rows, col + cols);
  if constexpr (Pattern == sparsity::dense) {
    const Eigen::Map<const matrix, Eigen::Aligned> A(a, rows, cols),
        A1(a1, rows, cols);
    if constexpr (Overwrite)
      O.block(row, col, rows, cols) = alpha * (A + A1);
    else
      O.block(row, col, rows, cols) += alpha * (A + A1);
  } else {
    auto dst = O.block(row, col, rows, rows).diagonal();
    const Eigen::Map<const vector, Eigen::Aligned> A(a, rows), A1(a1, rows);
    if constexpr (Overwrite)
      dst = alpha * (A + A1);
    else
      dst += alpha * (A + A1);
  }
}

#define MOTO_DUMP_WRAPPER(name, pattern, overwrite)                            \
  extern "C" __attribute__((visibility("default"))) void name(                 \
      const double *a, size_t rows, size_t cols, double *out, size_t out_rows, \
      size_t row_offset, size_t col_offset, double alpha) {                    \
    dump_panel<sparsity::pattern, overwrite>(a, rows, cols, out, out_rows,     \
                                             row_offset, col_offset, alpha);   \
  }
MOTO_DUMP_WRAPPER(moto_linear_dump_dense_accumulate, dense, false)
MOTO_DUMP_WRAPPER(moto_linear_dump_dense_overwrite, dense, true)
MOTO_DUMP_WRAPPER(moto_linear_dump_diag_accumulate, diag, false)
MOTO_DUMP_WRAPPER(moto_linear_dump_diag_overwrite, diag, true)
MOTO_DUMP_WRAPPER(moto_linear_dump_eye_accumulate, eye, false)
MOTO_DUMP_WRAPPER(moto_linear_dump_eye_overwrite, eye, true)
#undef MOTO_DUMP_WRAPPER

#define MOTO_DUMP_PAIR_WRAPPER(pattern, overwrite, mode)                       \
  extern "C" __attribute__((visibility("default"))) void                     \
      moto_linear_dump_pair_##pattern##_##mode(                               \
          const double *a, const double *a1, size_t rows, size_t cols,         \
          double *out, size_t out_rows, size_t row, size_t col, double alpha) {\
    dump_pair<sparsity::pattern, overwrite>(                                   \
        a, a1, rows, cols, out, out_rows, row, col, alpha);                    \
  }
MOTO_DUMP_PAIR_WRAPPER(dense, false, accumulate)
MOTO_DUMP_PAIR_WRAPPER(dense, true, overwrite)
MOTO_DUMP_PAIR_WRAPPER(diag, false, accumulate)
MOTO_DUMP_PAIR_WRAPPER(diag, true, overwrite)
#undef MOTO_DUMP_PAIR_WRAPPER

template <bool LhsTranspose, bool RhsTranspose>
void pair_dense_dense(const double *a, size_t ar, size_t ac, size_t ak,
                      const double *b, size_t br, size_t bc, size_t bk,
                      double *out, size_t out_rows, size_t out_row,
                      size_t out_col, size_t n, double alpha) {
  const Eigen::Map<const matrix, Eigen::Aligned> A(a, ar, ac);
  const Eigen::Map<const matrix, Eigen::Aligned> B(b, br, bc);
  Eigen::Map<matrix> O(out, out_rows, out_col + (RhsTranspose ? br : bc));
  auto block =
      O.block(out_row, out_col, LhsTranspose ? ac : ar, RhsTranspose ? br : bc);
  if constexpr (LhsTranspose && RhsTranspose)
    block.noalias() += alpha * A.middleRows(ak, n).transpose() *
                       B.middleCols(bk, n).transpose();
  else if constexpr (LhsTranspose)
    block.noalias() +=
        alpha * A.middleRows(ak, n).transpose() * B.middleRows(bk, n);
  else if constexpr (RhsTranspose)
    block.noalias() +=
        alpha * A.middleCols(ak, n) * B.middleCols(bk, n).transpose();
  else
    block.noalias() += alpha * A.middleCols(ak, n) * B.middleRows(bk, n);
}

template <bool Eye, bool RhsTranspose>
void pair_struct_dense(const double *a, size_t ak, const double *b, size_t br,
                       size_t bc, size_t bk, double *out, size_t out_rows,
                       size_t out_row, size_t out_col, size_t n, double alpha) {
  const Eigen::Map<const matrix, Eigen::Aligned> B(b, br, bc);
  Eigen::Map<matrix> O(out, out_rows, out_col + (RhsTranspose ? br : bc));
  auto target = O.block(out_row, out_col, n, RhsTranspose ? br : bc);
  if constexpr (RhsTranspose) {
    if constexpr (Eye)
      target += alpha * B.middleCols(bk, n).transpose();
    else
      target.noalias() += alpha *
                          Eigen::Map<const vector>(a + ak, n).asDiagonal() *
                          B.middleCols(bk, n).transpose();
  } else if constexpr (Eye) {
    target += alpha * B.middleRows(bk, n);
  } else {
    target.noalias() += alpha *
                        Eigen::Map<const vector>(a + ak, n).asDiagonal() *
                        B.middleRows(bk, n);
  }
}

template <bool LhsTranspose, bool Eye>
void pair_dense_struct(const double *a, size_t ar, size_t ac, size_t ak,
                       const double *b, size_t bk, double *out, size_t out_rows,
                       size_t out_row, size_t out_col, size_t n, double alpha) {
  const Eigen::Map<const matrix, Eigen::Aligned> A(a, ar, ac);
  Eigen::Map<matrix> O(out, out_rows, out_col + n);
  auto target = O.block(out_row, out_col, LhsTranspose ? ac : ar, n);
  if constexpr (LhsTranspose) {
    if constexpr (Eye)
      target += alpha * A.middleRows(ak, n).transpose();
    else
      target.noalias() += alpha * A.middleRows(ak, n).transpose() *
                          Eigen::Map<const vector>(b + bk, n).asDiagonal();
  } else if constexpr (Eye) {
    target += alpha * A.middleCols(ak, n);
  } else {
    target.noalias() += alpha * A.middleCols(ak, n) *
                        Eigen::Map<const vector>(b + bk, n).asDiagonal();
  }
}

template <bool LhsEye, bool RhsEye>
void pair_struct_struct(const double *a, size_t ak, const double *b, size_t bk,
                        double *out, size_t out_rows, size_t out_row,
                        size_t out_col, size_t n, double alpha) {
  Eigen::Map<matrix> O(out, out_rows, out_col + n);
  auto diagonal = O.block(out_row, out_col, n, n).diagonal();
  if constexpr (LhsEye && RhsEye)
    diagonal.array() += alpha;
  else if constexpr (LhsEye)
    diagonal.array() += alpha * Eigen::Map<const vector>(b + bk, n).array();
  else if constexpr (RhsEye)
    diagonal.array() += alpha * Eigen::Map<const vector>(a + ak, n).array();
  else
    diagonal.array() += alpha * Eigen::Map<const vector>(a + ak, n).array() *
                        Eigen::Map<const vector>(b + bk, n).array();
}

#define MOTO_PAIR_WRAPPER(name, ...)                                           \
  extern "C" __attribute__((visibility("default"))) void name(                 \
      const double *a, size_t ar, size_t ac, size_t ak, const double *b,       \
      size_t br, size_t bc, size_t bk, double *out, size_t out_rows,           \
      size_t out_row, size_t out_col, size_t n, double alpha) {                \
    __VA_ARGS__;                                                               \
  }
#define MOTO_PAIR_DD(tag, lt, rt)                                              \
  MOTO_PAIR_WRAPPER(moto_linear_pair_dense_dense_##tag,                        \
                    pair_dense_dense<lt, rt>(a, ar, ac, ak, b, br, bc, bk,     \
                                             out, out_rows, out_row, out_col,  \
                                             n, alpha))
MOTO_PAIR_DD(nn, false, false)
MOTO_PAIR_DD(tn, true, false)
MOTO_PAIR_DD(nt, false, true)
MOTO_PAIR_DD(tt, true, true)
#undef MOTO_PAIR_DD
#define MOTO_PAIR_SD(lp, le, tag, rt)                                          \
  MOTO_PAIR_WRAPPER(moto_linear_pair_##lp##_dense_##tag,                       \
                    pair_struct_dense<le, rt>(a, ak, b, br, bc, bk, out,       \
                                              out_rows, out_row, out_col, n,   \
                                              alpha))
MOTO_PAIR_SD(diag, false, n, false)
MOTO_PAIR_SD(diag, false, t, true)
MOTO_PAIR_SD(eye, true, n, false)
MOTO_PAIR_SD(eye, true, t, true)
#undef MOTO_PAIR_SD
#define MOTO_PAIR_DS(tag, lt, rp, re)                                          \
  MOTO_PAIR_WRAPPER(moto_linear_pair_dense_##rp##_##tag,                       \
                    pair_dense_struct<lt, re>(a, ar, ac, ak, b, bk, out,       \
                                              out_rows, out_row, out_col, n,   \
                                              alpha))
MOTO_PAIR_DS(n, false, diag, false)
MOTO_PAIR_DS(t, true, diag, false)
MOTO_PAIR_DS(n, false, eye, true)
MOTO_PAIR_DS(t, true, eye, true)
#undef MOTO_PAIR_DS
#define MOTO_PAIR_SS(lp, le, rp, re)                                           \
  MOTO_PAIR_WRAPPER(moto_linear_pair_##lp##_##rp,                              \
                    pair_struct_struct<le, re>(a, ak, b, bk, out, out_rows,    \
                                               out_row, out_col, n, alpha))
MOTO_PAIR_SS(diag, false, diag, false)
MOTO_PAIR_SS(diag, false, eye, true)
MOTO_PAIR_SS(eye, true, diag, false)
MOTO_PAIR_SS(eye, true, eye, true)
#undef MOTO_PAIR_SS
#undef MOTO_PAIR_WRAPPER

extern "C"
    __attribute__((visibility("default"))) void moto_linear_zero(double *out,
                                                                 size_t n) {
  Eigen::Map<vector>(out, n).setZero();
}

extern "C" __attribute__((visibility("default"))) void
moto_linear_jdx_dense(const double *jac, size_t rows, size_t cols,
                      const double *x, size_t x_offset, double *out,
                      size_t out_offset) {
  Eigen::Map<vector>(out + out_offset, rows).noalias() +=
      Eigen::Map<const matrix, Eigen::Aligned>(jac, rows, cols) *
      Eigen::Map<const vector>(x + x_offset, cols);
}

extern "C" __attribute__((visibility("default"))) void
moto_linear_jdx_diag(const double *jac, size_t n, const double *x,
                     size_t x_offset, double *out, size_t out_offset) {
  Eigen::Map<vector>(out + out_offset, n).array() +=
      Eigen::Map<const vector, Eigen::Aligned>(jac, n).array() *
      Eigen::Map<const vector>(x + x_offset, n).array();
}

extern "C" __attribute__((visibility("default"))) void
moto_linear_jdx_eye(size_t n, const double *x, size_t x_offset, double *out,
                    size_t out_offset) {
  Eigen::Map<vector>(out + out_offset, n) +=
      Eigen::Map<const vector>(x + x_offset, n);
}

extern "C" __attribute__((visibility("default"))) void
moto_linear_axpy(size_t n, double alpha, const double *x, double *out) {
  Eigen::Map<vector>(out, n).noalias() +=
      alpha * Eigen::Map<const vector>(x, n);
}

extern "C" __attribute__((visibility("default"))) void
moto_linear_cond_grad_dense(const double *jac, size_t rows, size_t cols,
                            const double *residual, size_t row_offset,
                            double *g, size_t col_offset) {
  Eigen::Map<vector>(g + col_offset, cols).noalias() +=
      Eigen::Map<const matrix, Eigen::Aligned>(jac, rows, cols).transpose() *
      Eigen::Map<const vector>(residual + row_offset, rows);
}

extern "C" __attribute__((visibility("default"))) void
moto_linear_cond_grad_diag(const double *jac, size_t n, const double *residual,
                           size_t row_offset, double *g, size_t col_offset) {
  Eigen::Map<vector>(g + col_offset, n).array() +=
      Eigen::Map<const vector, Eigen::Aligned>(jac, n).array() *
      Eigen::Map<const vector>(residual + row_offset, n).array();
}

extern "C" __attribute__((visibility("default"))) void
moto_linear_cond_grad_eye(size_t n, const double *residual, size_t row_offset,
                          double *g, size_t col_offset) {
  Eigen::Map<vector>(g + col_offset, n) +=
      Eigen::Map<const vector>(residual + row_offset, n);
}

extern "C" __attribute__((visibility("default"))) void
moto_linear_cond_hess_dense_dense(const double *lhs, size_t lhs_rows,
                                  size_t lhs_cols, size_t lhs_offset,
                                  const double *rhs, size_t rhs_rows,
                                  size_t rhs_cols, size_t rhs_offset,
                                  const double *weight, size_t weight_offset,
                                  size_t n, double *out) {
  Eigen::Map<matrix, Eigen::Aligned>(out, lhs_cols, rhs_cols).noalias() +=
      Eigen::Map<const matrix, Eigen::Aligned>(lhs, lhs_rows, lhs_cols)
          .middleRows(lhs_offset, n)
          .transpose() *
      Eigen::Map<const vector>(weight + weight_offset, n).asDiagonal() *
      Eigen::Map<const matrix, Eigen::Aligned>(rhs, rhs_rows, rhs_cols)
          .middleRows(rhs_offset, n);
}

template <bool Eye>
void cond_hess_dense_struct(const double *dense, size_t dense_rows,
                            size_t dense_cols, size_t dense_offset,
                            const double *structured, size_t struct_offset,
                            const double *weight, size_t weight_offset,
                            size_t n, double *out) {
  vector scale = Eigen::Map<const vector>(weight + weight_offset, n);
  if constexpr (!Eye)
    scale.array() *=
        Eigen::Map<const vector>(structured + struct_offset, n).array();
  Eigen::Map<matrix, Eigen::Aligned>(out, dense_cols, n).noalias() +=
      Eigen::Map<const matrix, Eigen::Aligned>(dense, dense_rows, dense_cols)
          .middleRows(dense_offset, n)
          .transpose() *
      scale.asDiagonal();
}

template <bool Eye>
void cond_hess_struct_dense(const double *structured, size_t struct_offset,
                            const double *dense, size_t dense_rows,
                            size_t dense_cols, size_t dense_offset,
                            const double *weight, size_t weight_offset,
                            size_t n, double *out) {
  vector scale = Eigen::Map<const vector>(weight + weight_offset, n);
  if constexpr (!Eye)
    scale.array() *=
        Eigen::Map<const vector>(structured + struct_offset, n).array();
  Eigen::Map<matrix, Eigen::Aligned>(out, n, dense_cols).noalias() +=
      scale.asDiagonal() *
      Eigen::Map<const matrix, Eigen::Aligned>(dense, dense_rows, dense_cols)
          .middleRows(dense_offset, n);
}

template <bool LhsEye, bool RhsEye>
void cond_hess_struct_struct(const double *lhs, size_t lhs_offset,
                             const double *rhs, size_t rhs_offset,
                             const double *weight, size_t weight_offset,
                             size_t n, double *out) {
  vector scale = Eigen::Map<const vector>(weight + weight_offset, n);
  if constexpr (!LhsEye)
    scale.array() *= Eigen::Map<const vector>(lhs + lhs_offset, n).array();
  if constexpr (!RhsEye)
    scale.array() *= Eigen::Map<const vector>(rhs + rhs_offset, n).array();
  Eigen::Map<vector, Eigen::Aligned>(out, n) += scale;
}

#define MOTO_COND_WRAPPER(name, ...)                                           \
  extern "C" __attribute__((visibility("default"))) void name(                 \
      const double *a, size_t ar, size_t ac, size_t ao, const double *b,       \
      size_t br, size_t bc, size_t bo, const double *w, size_t wo, size_t n,   \
      double *out) {                                                           \
    __VA_ARGS__;                                                               \
  }
MOTO_COND_WRAPPER(moto_linear_cond_hess_dense_diag,
                  cond_hess_dense_struct<false>(a, ar, ac, ao, b, bo, w, wo, n,
                                                out))
MOTO_COND_WRAPPER(moto_linear_cond_hess_dense_eye,
                  cond_hess_dense_struct<true>(a, ar, ac, ao, b, bo, w, wo, n,
                                               out))
MOTO_COND_WRAPPER(moto_linear_cond_hess_diag_dense,
                  cond_hess_struct_dense<false>(a, ao, b, br, bc, bo, w, wo, n,
                                                out))
MOTO_COND_WRAPPER(moto_linear_cond_hess_eye_dense,
                  cond_hess_struct_dense<true>(a, ao, b, br, bc, bo, w, wo, n,
                                               out))
MOTO_COND_WRAPPER(moto_linear_cond_hess_diag_diag,
                  cond_hess_struct_struct<false, false>(a, ao, b, bo, w, wo, n,
                                                        out))
MOTO_COND_WRAPPER(moto_linear_cond_hess_diag_eye,
                  cond_hess_struct_struct<false, true>(a, ao, b, bo, w, wo, n,
                                                       out))
MOTO_COND_WRAPPER(moto_linear_cond_hess_eye_diag,
                  cond_hess_struct_struct<true, false>(a, ao, b, bo, w, wo, n,
                                                       out))
MOTO_COND_WRAPPER(moto_linear_cond_hess_eye_eye,
                  cond_hess_struct_struct<true, true>(a, ao, b, bo, w, wo, n,
                                                      out))
#undef MOTO_COND_WRAPPER

void product_spec::validate() const {
  if (!sparse.rows || !sparse.cols || !other_rows || !other_cols || !out_rows ||
      !out_cols)
    throw std::invalid_argument("empty sparse JIT product dimension");
  for (const auto &p : sparse.panels) {
    if (p.pattern == sparsity::unknown || p.pattern == sparsity::num ||
        p.row_offset + p.rows > sparse.rows ||
        p.col_offset + p.cols > sparse.cols)
      throw std::invalid_argument("invalid sparse JIT panel");
  }
}

void product_kernel::operator()(std::span<scalar_t *> pointers) const {
  if (!function_ || pointers.size() != pointer_count())
    throw std::invalid_argument("invalid sparse JIT product invocation");
  function_(pointers.data(), pointers[other_slot()], pointers[output_slot()]);
}

void product_kernel::operator()(std::span<scalar_t *> panels,
                                const scalar_t *other, scalar_t *output) const {
  if (!function_ || panels.size() != spec_.sparse.panels.size())
    throw std::invalid_argument("invalid sparse JIT product invocation");
  function_(panels.data(), other, output);
}

std::string emit_product_function(const product_spec &spec,
                                  std::string_view name, bool exported) {
  spec.validate();
  std::ostringstream s;
  const auto op_name = [&] {
    return spec.op == product_op::times             ? "times"
           : spec.op == product_op::transpose_times ? "transpose_times"
           : spec.op == product_op::right_times     ? "right_times"
                                                     : "right_transpose_times";
  };
  const auto kernel_name = [&](sparsity pattern) {
    const char *prefix = pattern == sparsity::dense  ? "moto_linear_dense_"
                         : pattern == sparsity::diag ? "moto_linear_diag_"
                                                     : "moto_linear_eye_";
    return std::string(prefix) + op_name();
  };
  const auto fused = fused_panel_groups(spec.sparse);
  s << "#include <cstddef>\n";
  for (const auto &panel : spec.sparse.panels)
    s << "extern \"C\" void " << kernel_name(panel.pattern)
      << "(const double*,std::size_t,std::size_t,const double*,std::size_t,"
         "std::size_t,double*,std::size_t,std::size_t,std::size_t,std::size_t,"
         "double);\n";
  for (const auto &group : fused) {
    const auto pattern = spec.sparse.panels[group.front()].pattern;
    if (pattern == sparsity::eye) continue;
    s << "extern \"C\" void moto_linear_fused_"
      << (pattern == sparsity::dense ? "dense_" : "diag_")
      << op_name()
      << "(const double*,const double*,std::size_t,std::size_t,const double*,"
         "std::size_t,std::size_t,double*,std::size_t,std::size_t,std::size_t,"
         "std::size_t,double);\n";
  }
  if (exported)
    s << "extern \"C\" __attribute__((visibility(\"default\"))) ";
  else
    s << "static ";
  s << "void " << name
    << "(double *const *p, const double *__restrict b, "
       "double *__restrict o) {\n";
  for (size_t i = 0; i < spec.sparse.panels.size(); ++i)
    s << "  const double *__restrict a" << i << " = p[" << i << "];\n";
  s << std::setprecision(std::numeric_limits<scalar_t>::max_digits10);

  std::vector<int> fused_owner(spec.sparse.panels.size(), -1);
  for (size_t i = 0; i < fused.size(); ++i)
    for (const auto panel : fused[i])
      fused_owner[panel] = static_cast<int>(i);
  for (size_t pi = 0; pi < spec.sparse.panels.size(); ++pi) {
    if (fused_owner[pi] >= 0) {
      const auto &group = fused[static_cast<size_t>(fused_owner[pi])];
      if (pi != group.front()) continue;
      const auto &a = spec.sparse.panels[pi];
      if (a.pattern == sparsity::eye) {
        const auto scale = spec.sign * static_cast<scalar_t>(group.size());
        s << "  " << kernel_name(a.pattern) << "(a" << pi << ',' << a.rows
          << ',' << a.cols << ",b," << spec.other_rows << ','
          << spec.other_cols << ",o," << spec.out_rows << ',' << spec.out_cols
          << ',' << a.row_offset << ',' << a.col_offset << ',' << scale
          << ");\n";
      } else {
        const auto prefix = a.pattern == sparsity::dense ? "dense_" : "diag_";
        s << "  moto_linear_fused_" << prefix << op_name()
          << "(a" << group[0] << ",a" << group[1] << ',' << a.rows << ','
          << a.cols << ",b," << spec.other_rows << ',' << spec.other_cols
          << ",o," << spec.out_rows << ',' << spec.out_cols << ','
          << a.row_offset << ',' << a.col_offset << ',' << spec.sign << ");\n";
        for (size_t gi = 2; gi < group.size(); ++gi)
          s << "  " << kernel_name(a.pattern) << "(a" << group[gi] << ','
            << a.rows << ',' << a.cols << ",b," << spec.other_rows << ','
            << spec.other_cols << ",o," << spec.out_rows << ',' << spec.out_cols
            << ',' << a.row_offset << ',' << a.col_offset << ',' << spec.sign
            << ");\n";
      }
      continue;
    }
    const auto &a = spec.sparse.panels[pi];
    s << "  " << kernel_name(a.pattern) << "(a" << pi << ',' << a.rows << ','
      << a.cols << ",b," << spec.other_rows << ',' << spec.other_cols << ",o,"
      << spec.out_rows << ',' << spec.out_cols << ',' << a.row_offset << ','
      << a.col_offset << ',' << spec.sign << ");\n";
  }
  s << "}\n";
  return s.str();
}

std::string emit_product_source(const product_spec &spec) {
  return emit_product_function(spec, symbol_name, true);
}

product_kernel compile_product(product_spec spec,
                               const std::filesystem::path &cache_dir) {
  auto function = reinterpret_cast<product_kernel::function_type>(
      compile_source(emit_product_source(spec), cache_dir));
  return product_kernel(std::move(spec), function);
}

void batch_product_spec::validate() const {
  if (products.empty())
    throw std::invalid_argument("empty sparse JIT batch");
  for (const auto &product : products)
    product.validate();
}

void batch_product_kernel::operator()(std::span<scalar_t *> pointers) const {
  if (!function_ || pointers.size() != pointers_)
    throw std::invalid_argument("invalid sparse JIT batch invocation");
  function_(pointers.data());
}

batch_product_kernel
compile_batch_product(batch_product_spec spec,
                      const std::filesystem::path &cache_dir) {
  spec.validate();
  std::ostringstream signature;
  for (const auto &product : spec.products) {
    signature << static_cast<int>(product.op) << ',' << product.other_rows
              << ',' << product.other_cols << ',' << product.out_rows << ','
              << product.out_cols << ',' << product.sign << ';';
    for (const auto &panel : product.sparse.panels)
      signature << static_cast<int>(panel.pattern) << ',' << panel.row_offset
                << ',' << panel.col_offset << ',' << panel.rows << ','
                << panel.cols << ';';
    signature << '|';
  }
  const auto key = signature.str();
  {
    std::lock_guard lock(batch_mutex);
    if (auto found = batch_kernels.find(key); found != batch_kernels.end())
      return found->second;
  }
  std::ostringstream source;
  size_t offset = 0;
  for (size_t i = 0; i < spec.products.size(); ++i) {
    source << emit_product_function(
        spec.products[i], "moto_linear_jit_op_" + std::to_string(i), false);
  }
  source << "extern \"C\" __attribute__((visibility(\"default\"))) void "
         << symbol_name << "(double *const *p) {\n";
  for (size_t i = 0; i < spec.products.size(); ++i) {
    const size_t panels = spec.products[i].sparse.panels.size();
    source << "  moto_linear_jit_op_" << i << "(p+" << offset << ",p["
           << offset + panels << "],p[" << offset + panels + 1 << "]);\n";
    offset += panels + 2;
  }
  source << "}\n";
  auto function = reinterpret_cast<batch_product_kernel::function_type>(
      compile_source(source.str(), cache_dir));
  auto kernel = batch_product_kernel(offset, function);
  std::lock_guard lock(batch_mutex);
  return batch_kernels.emplace(key, kernel).first->second;
}

struct cached_product {
  product_op op;
  scalar_t sign;
  size_t other_rows;
  size_t other_cols;
  size_t out_rows;
  size_t out_cols;
  product_kernel kernel;
  std::vector<scalar_t *> pointers;
};

struct cached_sparse_product {
  const sparse_matrix *other;
  product_op op;
  scalar_t sign;
  size_t out_rows;
  size_t out_cols;
  batch_product_kernel kernel;
  std::vector<scalar_t *> pointers;
};

struct simple_key {
  enum class kind { weighted_gram, dense_write } op;
  size_t out_rows = 0;
  scalar_t alpha = 0.;
  bool overwrite = false;
  bool operator==(const simple_key &) const = default;
};

struct cached_simple_kernel {
  simple_key key;
  batch_product_kernel kernel;
  std::vector<scalar_t *> pointers;
};

struct matrix_cache {
  std::mutex mutex;
  std::vector<cached_product> products;
  std::vector<cached_sparse_product> sparse_products;
  std::vector<cached_simple_kernel> simple;
};

matrix_layout describe(const ::moto::sparse_matrix &sparse) {
  matrix_layout layout{.rows = sparse.rows_, .cols = sparse.cols_};
  const auto add = [&](const auto &panels, sparsity pattern) {
    for (const auto &p : panels)
      layout.panels.push_back({pattern, static_cast<size_t>(p.row_st_),
                               static_cast<size_t>(p.col_st_),
                               static_cast<size_t>(p.rows_),
                               static_cast<size_t>(p.cols_)});
  };
  add(sparse.dense_panels_, sparsity::dense);
  add(sparse.diag_panels_, sparsity::diag);
  add(sparse.eye_panels_, sparsity::eye);
  return layout;
}

std::vector<scalar_t *> panel_pointers(const ::moto::sparse_matrix &sparse) {
  std::vector<scalar_t *> pointers;
  const auto add = [&](const auto &panels) {
    for (const auto &p : panels)
      pointers.push_back(const_cast<scalar_t *>(p.data_.data()));
  };
  add(sparse.dense_panels_);
  add(sparse.diag_panels_);
  add(sparse.eye_panels_);
  return pointers;
}

void run_product(const ::moto::sparse_matrix &sparse, product_op op,
                 scalar_t sign, const scalar_t *other, size_t other_rows,
                 size_t other_cols, scalar_t *out, size_t out_rows,
                 size_t out_cols) {
  if (!sparse.jit_cache_)
    sparse.jit_cache_ = std::make_shared<matrix_cache>();
  auto &cache = *sparse.jit_cache_;
  const auto matches = [&](const auto &entry) {
    return entry.op == op && entry.sign == sign &&
           entry.other_rows == other_rows && entry.other_cols == other_cols &&
           entry.out_rows == out_rows && entry.out_cols == out_cols;
  };
  auto found = std::ranges::find_if(cache.products, matches);
  if (found != cache.products.end()) {
    if (out)
      found->kernel(found->pointers, other, out);
    return;
  }

  std::lock_guard lock(cache.mutex);
  found = std::ranges::find_if(cache.products, matches);
  if (found != cache.products.end()) {
    if (out)
      found->kernel(found->pointers, other, out);
    return;
  }

  auto layout = describe(sparse);
  auto pointers = panel_pointers(sparse);
  product_spec spec{.sparse = std::move(layout),
                    .op = op,
                    .other_rows = other_rows,
                    .other_cols = other_cols,
                    .out_rows = out_rows,
                    .out_cols = out_cols,
                    .sign = sign};
  auto kernel = compile_product(std::move(spec));
  if (out)
    kernel(pointers, other, out);
  cache.products.push_back({op, sign, other_rows, other_cols, out_rows,
                            out_cols, std::move(kernel), std::move(pointers)});
}

void prepare_product(const sparse_matrix &sparse, product_op op, scalar_t sign,
                     size_t other_rows, size_t other_cols, size_t out_rows,
                     size_t out_cols) {
  if (sparse.is_empty() || !other_rows || !other_cols || !out_rows || !out_cols)
    return;
  run_product(sparse, op, sign, nullptr, other_rows, other_cols, nullptr,
              out_rows, out_cols);
}

void prepare_products(std::span<const product_request> requests) {
  struct pending {
    matrix_cache *cache;
    product_spec spec;
    cached_product entry;
  };
  std::vector<pending> missing;
  for (const auto &r : requests) {
    if (!r.sparse || r.sparse->is_empty() || !r.other_rows || !r.other_cols ||
        !r.out_rows || !r.out_cols)
      continue;
    if (!r.sparse->jit_cache_)
      r.sparse->jit_cache_ = std::make_shared<matrix_cache>();
    auto &cache = *r.sparse->jit_cache_;
    const auto matches = [&](const auto &entry) {
      return entry.op == r.op && entry.sign == r.sign &&
             entry.other_rows == r.other_rows &&
             entry.other_cols == r.other_cols && entry.out_rows == r.out_rows &&
             entry.out_cols == r.out_cols;
    };
    std::lock_guard lock(cache.mutex);
    if (std::ranges::find_if(cache.products, matches) != cache.products.end())
      continue;
    auto spec = product_spec{.sparse = describe(*r.sparse),
                             .op = r.op,
                             .other_rows = r.other_rows,
                             .other_cols = r.other_cols,
                             .out_rows = r.out_rows,
                             .out_cols = r.out_cols,
                             .sign = r.sign};
    missing.push_back({&cache,
                       spec,
                       {r.op,
                        r.sign,
                        r.other_rows,
                        r.other_cols,
                        r.out_rows,
                        r.out_cols,
                        {},
                        panel_pointers(*r.sparse)}});
  }
  if (missing.empty())
    return;

  std::ostringstream source;
  for (size_t i = 0; i < missing.size(); ++i) {
    source << emit_product_function(
        missing[i].spec, "moto_linear_jit_precompiled_" + std::to_string(i),
        false);
  }
  source << "extern \"C\" __attribute__((visibility(\"default\"))) void *"
         << symbol_name << "(std::size_t i) { static void *f[]={";
  for (size_t i = 0; i < missing.size(); ++i) {
    if (i)
      source << ',';
    source << "reinterpret_cast<void*>(&moto_linear_jit_precompiled_" << i
           << ')';
  }
  source << "}; return f[i]; }\n";
  using registry_type = void *(*)(size_t);
  auto registry = reinterpret_cast<registry_type>(
      compile_source(source.str(), "gen/linear_backend"));
  for (size_t i = 0; i < missing.size(); ++i) {
    auto &item = missing[i];
    item.entry.kernel = product_kernel(
        std::move(item.spec),
        reinterpret_cast<product_kernel::function_type>(registry(i)));
    std::lock_guard lock(item.cache->mutex);
    item.cache->products.push_back(std::move(item.entry));
  }
}

namespace {
struct effective_panel {
  panel_layout panel;
  bool transpose;
  size_t row() const { return transpose ? panel.col_offset : panel.row_offset; }
  size_t col() const { return transpose ? panel.row_offset : panel.col_offset; }
  size_t rows() const { return transpose ? panel.cols : panel.rows; }
  size_t cols() const { return transpose ? panel.rows : panel.cols; }
  bool structured() const { return panel.pattern != sparsity::dense; }
};

std::string emit_sparse_product_source(const matrix_layout &lhs,
                                       const matrix_layout &rhs,
                                       bool lhs_transpose, bool rhs_transpose,
                                       scalar_t sign, size_t out_rows) {
  std::ostringstream s;
  const auto pattern_name = [](sparsity p) {
    return p == sparsity::dense  ? "dense"
           : p == sparsity::diag ? "diag"
                                 : "eye";
  };
  const auto helper_name = [&](const effective_panel &l,
                               const effective_panel &r) {
    std::string name = "moto_linear_pair_";
    name += pattern_name(l.panel.pattern);
    name += '_';
    name += pattern_name(r.panel.pattern);
    if (!l.structured() || !r.structured()) {
      name += '_';
      if (!l.structured())
        name += l.transpose ? 't' : 'n';
      if (!r.structured())
        name += r.transpose ? 't' : 'n';
    }
    return name;
  };
  s << "#include <cstddef>\n";
  for (const auto &lp : lhs.panels)
    for (const auto &rp : rhs.panels) {
      const effective_panel l{lp, lhs_transpose}, r{rp, rhs_transpose};
      s << "extern \"C\" void " << helper_name(l, r)
        << "(const double*,std::size_t,std::size_t,std::size_t,const "
           "double*,std::size_t,std::size_t,std::size_t,double*,std::size_t,"
           "std::size_t,std::size_t,std::size_t,double);\n";
    }
  s << "extern \"C\" __attribute__((visibility(\"default\"))) void "
    << symbol_name << "(double *const *p) {\n";
  for (size_t li = 0; li < lhs.panels.size(); ++li) {
    const effective_panel l{lhs.panels[li], lhs_transpose};
    for (size_t ri = 0; ri < rhs.panels.size(); ++ri) {
      const effective_panel r{rhs.panels[ri], rhs_transpose};
      const size_t begin = std::max(l.col(), r.row());
      const size_t end = std::min(l.col() + l.cols(), r.row() + r.rows());
      if (end <= begin)
        continue;
      const size_t n = end - begin;
      const size_t lk = begin - l.col(), rk = begin - r.row();
      const size_t rslot = lhs.panels.size() + ri;
      const size_t oslot = lhs.panels.size() + rhs.panels.size();
      const size_t out_row = l.row() + (l.structured() ? lk : 0);
      const size_t out_col = r.col() + (r.structured() ? rk : 0);
      s << "  " << helper_name(l, r) << "(p[" << li << "]," << l.panel.rows
        << ',' << l.panel.cols << ',' << lk << ",p[" << rslot << "],"
        << r.panel.rows << ',' << r.panel.cols << ',' << rk << ",p[" << oslot
        << "]," << out_rows << ',' << out_row << ',' << out_col << ',' << n
        << ',' << sign << ");\n";
    }
  }
  return s.str() + "}\n";
}
} // namespace

void run_sparse_product(const sparse_matrix &sparse, const sparse_matrix &other,
                        product_op op, scalar_t sign, scalar_t *out,
                        size_t out_rows, size_t out_cols) {
  if (!sparse.jit_cache_)
    sparse.jit_cache_ = std::make_shared<matrix_cache>();
  auto &cache = *sparse.jit_cache_;
  const auto matches = [&](const auto &entry) {
    return entry.other == &other && entry.op == op && entry.sign == sign &&
           entry.out_rows == out_rows && entry.out_cols == out_cols;
  };
  if (auto found = std::ranges::find_if(cache.sparse_products, matches);
      found != cache.sparse_products.end()) {
    if (out) {
      found->pointers.back() = out;
      found->kernel(found->pointers);
    }
    return;
  }
  std::lock_guard lock(cache.mutex);
  const bool lhs_t = op == product_op::transpose_times ||
                     op == product_op::right_transpose_times;
  const bool swap =
      op == product_op::right_times || op == product_op::right_transpose_times;
  auto a = describe(swap ? other : sparse);
  auto b = describe(swap ? sparse : other);
  auto pointers = panel_pointers(swap ? other : sparse);
  auto rhs_pointers = panel_pointers(swap ? sparse : other);
  pointers.insert(pointers.end(), rhs_pointers.begin(), rhs_pointers.end());
  pointers.push_back(out);
  const bool at = swap ? lhs_t : lhs_t;
  const bool bt = false;
  const auto source = emit_sparse_product_source(a, b, at, bt, sign, out_rows);
  auto function = reinterpret_cast<batch_product_kernel::function_type>(
      compile_source(source, "gen/linear_backend"));
  batch_product_kernel kernel(pointers.size(), function);
  if (out)
    kernel(pointers);
  cache.sparse_products.push_back({&other, op, sign, out_rows, out_cols,
                                   std::move(kernel), std::move(pointers)});
}

void prepare_sparse_product(const sparse_matrix &sparse,
                            const sparse_matrix &other, product_op op,
                            scalar_t sign, size_t out_rows, size_t out_cols) {
  if (sparse.is_empty() || other.is_empty() || !out_rows || !out_cols)
    return;
  run_sparse_product(sparse, other, op, sign, nullptr, out_rows, out_cols);
}

void run_dense_write(const sparse_matrix &sparse, scalar_t *out,
                     size_t out_rows, scalar_t alpha, bool overwrite) {
  if (!sparse.jit_cache_)
    sparse.jit_cache_ = std::make_shared<matrix_cache>();
  auto &cache = *sparse.jit_cache_;
  const simple_key key{simple_key::kind::dense_write, out_rows, alpha,
                       overwrite};
  if (auto found =
          std::ranges::find(cache.simple, key, &cached_simple_kernel::key);
      found != cache.simple.end()) {
    if (out) {
      found->pointers.back() = out;
      found->kernel(found->pointers);
    }
    return;
  }
  std::lock_guard lock(cache.mutex);
  auto layout = describe(sparse);
  const auto fused = fused_panel_groups(layout);
  auto pointers = panel_pointers(sparse);
  pointers.push_back(out);
  std::ostringstream source;
  source << "#include <cstddef>\n";
  for (const auto pattern : {"dense", "diag", "eye"})
    source << "extern \"C\" void moto_linear_dump_" << pattern << '_'
           << (overwrite ? "overwrite" : "accumulate")
           << "(const "
              "double*,std::size_t,std::size_t,double*,std::size_t,std::size_t,"
              "std::size_t,double);\n";
  if (overwrite)
    for (const auto pattern : {"dense", "diag"})
      source << "extern \"C\" void moto_linear_dump_" << pattern
             << "_accumulate(const double*,std::size_t,std::size_t,double*,"
                "std::size_t,std::size_t,std::size_t,double);\n";
  for (const auto pattern : {"dense", "diag"})
    source << "extern \"C\" void moto_linear_dump_pair_" << pattern << '_'
           << (overwrite ? "overwrite" : "accumulate")
           << "(const double*,const double*,std::size_t,std::size_t,double*,"
              "std::size_t,std::size_t,std::size_t,double);\n";
  source << "extern \"C\" __attribute__((visibility(\"default\"))) void "
         << symbol_name << "(double *const *p) {\n";
  std::vector<int> fused_owner(layout.panels.size(), -1);
  for (size_t gi = 0; gi < fused.size(); ++gi)
    for (const auto panel : fused[gi])
      fused_owner[panel] = static_cast<int>(gi);
  for (size_t i = 0; i < layout.panels.size(); ++i) {
    if (fused_owner[i] >= 0) {
      const auto &group = fused[static_cast<size_t>(fused_owner[i])];
      if (i != group.front()) continue;
      const auto &a = layout.panels[i];
      const char *pattern = a.pattern == sparsity::dense  ? "dense"
                            : a.pattern == sparsity::diag ? "diag"
                                                          : "eye";
      if (a.pattern == sparsity::eye) {
        source << "  moto_linear_dump_eye_"
               << (overwrite ? "overwrite" : "accumulate") << "(p[" << i
               << "]," << a.rows << ',' << a.cols << ",p["
               << layout.panels.size() << "]," << out_rows << ','
               << a.row_offset << ',' << a.col_offset << ','
               << alpha * static_cast<scalar_t>(group.size()) << ");\n";
      } else {
        source << "  moto_linear_dump_pair_" << pattern << '_'
               << (overwrite ? "overwrite" : "accumulate") << "(p["
               << group[0] << "],p[" << group[1] << "]," << a.rows << ','
               << a.cols << ",p[" << layout.panels.size() << "]," << out_rows
               << ',' << a.row_offset << ',' << a.col_offset << ',' << alpha
               << ");\n";
        for (size_t j = 2; j < group.size(); ++j)
          source << "  moto_linear_dump_" << pattern << "_accumulate(p["
                 << group[j] << "]," << a.rows << ',' << a.cols << ",p["
                 << layout.panels.size() << "]," << out_rows << ','
                 << a.row_offset << ',' << a.col_offset << ',' << alpha
                 << ");\n";
      }
      continue;
    }
    const auto &a = layout.panels[i];
    const char *pattern = a.pattern == sparsity::dense  ? "dense"
                          : a.pattern == sparsity::diag ? "diag"
                                                        : "eye";
    source << "  moto_linear_dump_" << pattern << '_'
           << (overwrite ? "overwrite" : "accumulate") << "(p[" << i << "],"
           << a.rows << ',' << a.cols << ",p[" << layout.panels.size() << "],"
           << out_rows << ',' << a.row_offset << ',' << a.col_offset << ','
           << alpha << ");\n";
  }
  source << "}\n";
  auto function = reinterpret_cast<batch_product_kernel::function_type>(
      compile_source(source.str(), "gen/linear_backend"));
  batch_product_kernel kernel(pointers.size(), function);
  if (out)
    kernel(pointers);
  cache.simple.push_back({key, std::move(kernel), std::move(pointers)});
}

void prepare_dense_write(const sparse_matrix &sparse, size_t out_rows,
                         scalar_t alpha, bool overwrite) {
  if (sparse.is_empty() || !out_rows)
    return;
  run_dense_write(sparse, nullptr, out_rows, alpha, overwrite);
}

void run_weighted_gram(const sparse_matrix &sparse, const scalar_t *middle,
                       scalar_t *out) {
  if (!sparse.jit_cache_)
    sparse.jit_cache_ = std::make_shared<matrix_cache>();
  auto &cache = *sparse.jit_cache_;
  const simple_key key{simple_key::kind::weighted_gram};
  if (auto found =
          std::ranges::find(cache.simple, key, &cached_simple_kernel::key);
      found != cache.simple.end()) {
    if (out) {
      found->pointers[found->pointers.size() - 2] =
          const_cast<scalar_t *>(middle);
      found->pointers.back() = out;
      found->kernel(found->pointers);
    }
    return;
  }
  std::lock_guard lock(cache.mutex);
  const auto layout = describe(sparse);
  auto pointers = panel_pointers(sparse);
  pointers.push_back(const_cast<scalar_t *>(middle));
  pointers.push_back(out);
  product_spec left{.sparse = layout,
                    .op = product_op::right_times,
                    .other_rows = sparse.rows_,
                    .other_cols = sparse.rows_,
                    .out_rows = sparse.rows_,
                    .out_cols = sparse.cols_};
  product_spec right{.sparse = layout,
                     .op = product_op::transpose_times,
                     .other_rows = sparse.rows_,
                     .other_cols = sparse.cols_,
                     .out_rows = sparse.cols_,
                     .out_cols = sparse.cols_};
  std::ostringstream source;
  source << emit_product_function(left, "moto_linear_jit_left", false)
         << emit_product_function(right, "moto_linear_jit_right", false)
         << "extern \"C\" __attribute__((visibility(\"default\"))) void "
         << symbol_name << "(double *const *p) { alignas(64) double tmp["
         << sparse.rows_ * sparse.cols_ << "]={}; moto_linear_jit_left(p,p["
         << layout.panels.size() << "],tmp); moto_linear_jit_right(p,tmp,p["
         << layout.panels.size() + 1 << "]); }\n";
  auto function = reinterpret_cast<batch_product_kernel::function_type>(
      compile_source(source.str(), "gen/linear_backend"));
  batch_product_kernel kernel(pointers.size(), function);
  if (out)
    kernel(pointers);
  cache.simple.push_back({key, std::move(kernel), std::move(pointers)});
}

void prepare_weighted_gram(const sparse_matrix &sparse) {
  if (sparse.is_empty())
    return;
  run_weighted_gram(sparse, nullptr, nullptr);
}

void condensation_spec::validate() const {
  if (rows == 0 || argument_count() == 0 || residual_signs.empty())
    throw std::invalid_argument(
        "JIT condensation requires non-empty rows, Jacobians, and sides");
  if (!jacobians.empty() && !jac_cols.empty())
    throw std::invalid_argument(
        "JIT condensation accepts either dense columns or panel layouts");
  if (std::ranges::any_of(jac_cols, [](size_t n) { return n == 0; }) ||
      std::ranges::any_of(jacobians, [&](const panel_layout &p) {
        return !p.rows || !p.cols || p.row_offset + p.rows > rows ||
               p.pattern == sparsity::unknown || p.pattern == sparsity::num;
      }))
    throw std::invalid_argument(
        "JIT condensation Jacobian columns must be non-zero");
}

size_t condensation_spec::argument_count() const {
  return jacobians.empty() ? jac_cols.size() : jacobians.size();
}

panel_layout condensation_hessian_layout(const panel_layout &lhs,
                                         const panel_layout &rhs) {
  const bool ls = lhs.pattern != sparsity::dense;
  const bool rs = rhs.pattern != sparsity::dense;
  const size_t row = std::max(lhs.row_offset, rhs.row_offset);
  const size_t end =
      std::min(lhs.row_offset + lhs.rows, rhs.row_offset + rhs.rows);
  if (lhs.pattern == sparsity::unknown || rhs.pattern == sparsity::unknown ||
      end <= row)
    return {sparsity::unknown, 0, 0, 0, 0};
  const size_t n = end - row;
  return {ls && rs ? sparsity::diag : sparsity::dense,
          lhs.col_offset + (ls ? row - lhs.row_offset : 0),
          rhs.col_offset + (rs ? row - rhs.row_offset : 0), ls ? n : lhs.cols,
          rs ? n : rhs.cols};
}

std::string
emit_jacobian_product_function(size_t rows,
                               const std::vector<panel_layout> &jacobians,
                               std::string_view name, bool exported) {
  if (!rows || jacobians.empty())
    throw std::invalid_argument("empty JIT Jacobian product");
  std::ostringstream s;
  s << "#include <cstddef>\n"
       "extern \"C\" void moto_linear_zero(double*,std::size_t);\n"
       "extern \"C\" void moto_linear_jdx_dense(const "
       "double*,std::size_t,std::size_t,const "
       "double*,std::size_t,double*,std::size_t);\n"
       "extern \"C\" void moto_linear_jdx_diag(const double*,std::size_t,const "
       "double*,std::size_t,double*,std::size_t);\n"
       "extern \"C\" void moto_linear_jdx_eye(std::size_t,const "
       "double*,std::size_t,double*,std::size_t);\n";
  if (exported)
    s << "extern \"C\" __attribute__((visibility(\"default\"))) ";
  else
    s << "static inline __attribute__((always_inline)) ";
  s << "void " << name << "(double *const *p) {\n"
    << "  double *__restrict out=p[" << 2 * jacobians.size() << "];\n"
    << "  moto_linear_zero(out," << rows << ");\n";
  for (size_t i = 0; i < jacobians.size(); ++i) {
    const auto &j = jacobians[i];
    s << "  const double *__restrict j" << i << "=p[" << i << "];\n"
      << "  const double *__restrict x" << i << "=p[" << jacobians.size() + i
      << "];\n";
    if (j.pattern == sparsity::dense)
      s << "  moto_linear_jdx_dense(j" << i << ',' << j.rows << ',' << j.cols
        << ",x" << i << ',' << j.col_offset << ",out," << j.row_offset
        << ");\n";
    else if (j.pattern == sparsity::diag)
      s << "  moto_linear_jdx_diag(j" << i << ',' << j.rows << ",x" << i << ','
        << j.col_offset << ",out," << j.row_offset << ");\n";
    else
      s << "  moto_linear_jdx_eye(" << j.rows << ",x" << i << ','
        << j.col_offset << ",out," << j.row_offset << ");\n";
  }
  s << "}\n";
  return s.str();
}

batch_product_kernel compile_batch_jacobian_product(
    std::vector<std::pair<size_t, std::vector<panel_layout>>> products,
    const std::filesystem::path &cache_dir) {
  if (products.empty())
    throw std::invalid_argument("empty OCP Jacobian product batch");
  std::ostringstream signature;
  for (const auto &[rows, panels] : products) {
    signature << rows << ':';
    for (const auto &p : panels)
      signature << static_cast<int>(p.pattern) << ',' << p.row_offset << ','
                << p.col_offset << ',' << p.rows << ',' << p.cols << ';';
    signature << '|';
  }
  const auto key = signature.str();
  {
    std::lock_guard lock(batch_mutex);
    if (auto found = jacobian_product_kernels.find(key);
        found != jacobian_product_kernels.end())
      return found->second;
  }
  std::ostringstream source;
  size_t offset = 0;
  for (size_t i = 0; i < products.size(); ++i) {
    source << emit_jacobian_product_function(
        products[i].first, products[i].second,
        "moto_linear_jit_jv_" + std::to_string(i), false);
  }
  source << "extern \"C\" __attribute__((visibility(\"default\"))) void "
         << symbol_name << "(double *const *p) {\n";
  for (size_t i = 0; i < products.size(); ++i) {
    source << "  moto_linear_jit_jv_" << i << "(p+" << offset << ");\n";
    offset += 2 * products[i].second.size() + 1;
  }
  source << "}\n";
  auto function = reinterpret_cast<batch_product_kernel::function_type>(
      compile_source(source.str(), cache_dir));
  auto kernel = batch_product_kernel(offset, function);
  std::lock_guard lock(batch_mutex);
  return jacobian_product_kernels.emplace(key, kernel).first->second;
}

size_t condensation_kernel::jacobian_slot(size_t arg) const {
  if (arg >= spec_.argument_count())
    throw std::out_of_range("JIT condensation Jacobian argument index");
  return arg;
}

size_t condensation_kernel::residual_slot(size_t side) const {
  if (side >= spec_.residual_signs.size())
    throw std::out_of_range("JIT condensation residual side index");
  return spec_.argument_count() + side;
}

size_t condensation_kernel::weight_slot(size_t side) const {
  if (side >= spec_.residual_signs.size())
    throw std::out_of_range("JIT condensation weight side index");
  return spec_.argument_count() + spec_.residual_signs.size() + side;
}

size_t condensation_kernel::gradient_slot(size_t arg) const {
  if (arg >= spec_.argument_count())
    throw std::out_of_range("JIT condensation gradient argument index");
  return spec_.argument_count() + 2 * spec_.residual_signs.size() + arg;
}

size_t condensation_kernel::hessian_slot(size_t lhs, size_t rhs) const {
  const size_t n = spec_.argument_count();
  if (lhs >= n || rhs >= n)
    throw std::out_of_range("JIT condensation Hessian argument index");
  const auto pairs = condensation_pairs(spec_);
  const auto found = std::ranges::find(pairs, std::pair(lhs, rhs));
  if (found == pairs.end())
    throw std::out_of_range("inactive JIT condensation Hessian block");
  return 2 * n + 2 * spec_.residual_signs.size() +
         std::distance(pairs.begin(), found);
}

size_t condensation_kernel::pointer_count() const {
  const size_t n = spec_.argument_count();
  return 2 * n + 2 * spec_.residual_signs.size() +
         condensation_pairs(spec_).size();
}

void condensation_kernel::operator()(std::span<scalar_t *> pointers) const {
  if (!function_)
    throw std::runtime_error("empty JIT condensation kernel");
  if (pointers.size() != pointer_count())
    throw std::invalid_argument("JIT condensation pointer table size mismatch");
  function_(pointers.data());
}

std::string emit_condensation_function(const condensation_spec &spec,
                                       std::string_view name, bool exported) {
  spec.validate();
  const size_t nargs = spec.argument_count();
  const size_t nsides = spec.residual_signs.size();
  std::ostringstream s;
  s << "#include <cstddef>\n"
       "extern \"C\" void moto_linear_zero(double*,std::size_t);\n"
       "extern \"C\" void moto_linear_axpy(std::size_t,double,const "
       "double*,double*);\n"
       "extern \"C\" void moto_linear_cond_grad_dense(const "
       "double*,std::size_t,std::size_t,const "
       "double*,std::size_t,double*,std::size_t);\n"
       "extern \"C\" void moto_linear_cond_grad_diag(const "
       "double*,std::size_t,const double*,std::size_t,double*,std::size_t);\n"
       "extern \"C\" void moto_linear_cond_grad_eye(std::size_t,const "
       "double*,std::size_t,double*,std::size_t);\n"
       "extern \"C\" void moto_linear_cond_hess_dense_dense(const "
       "double*,std::size_t,std::size_t,std::size_t,const "
       "double*,std::size_t,std::size_t,std::size_t,const "
       "double*,std::size_t,std::size_t,double*);\n";
  for (const auto lhs : {"dense", "diag", "eye"})
    for (const auto rhs : {"dense", "diag", "eye"})
      if (!(std::string_view(lhs) == "dense" &&
            std::string_view(rhs) == "dense"))
        s << "extern \"C\" void moto_linear_cond_hess_" << lhs << '_' << rhs
          << "(const double*,std::size_t,std::size_t,std::size_t,const "
             "double*,std::size_t,std::size_t,std::size_t,const "
             "double*,std::size_t,std::size_t,double*);\n";
  s << '\n';
  if (exported)
    s << "extern \"C\" __attribute__((visibility(\"default\"))) ";
  else
    s << "static inline __attribute__((always_inline)) ";
  s << "void " << name << "(double *const *p) {\n";
  for (size_t i = 0; i < nargs; ++i)
    s << "  const double *__restrict j" << i << " = p[" << i << "];\n";
  for (size_t side = 0; side < nsides; ++side)
    s << "  const double *__restrict r" << side << " = p[" << nargs + side
      << "];\n";
  for (size_t side = 0; side < nsides; ++side)
    s << "  const double *__restrict w" << side << " = p["
      << nargs + nsides + side << "];\n";
  for (size_t i = 0; i < nargs; ++i)
    s << "  double *__restrict g" << i << " = p[" << 2 * nsides + nargs + i
      << "];\n";
  const auto pairs = condensation_pairs(spec);
  for (size_t k = 0; k < pairs.size(); ++k)
    s << "  double *__restrict h" << pairs[k].first << '_' << pairs[k].second
      << " = p[" << 2 * nargs + 2 * nsides + k << "];\n";

  s << "  alignas(64) double residual[" << spec.rows << "]={};\n"
    << "  alignas(64) double weight[" << spec.rows << "]={};\n";
  s << std::setprecision(std::numeric_limits<scalar_t>::max_digits10);
  for (size_t side = 0; side < nsides; ++side) {
    s << "  moto_linear_axpy(" << spec.rows << ',' << spec.residual_signs[side]
      << ",r" << side << ",residual);\n"
      << "  moto_linear_axpy(" << spec.rows << ",1,w" << side << ",weight);\n";
  }

  if (!spec.jacobians.empty()) {
    for (size_t i = 0; i < nargs; ++i) {
      const auto &j = spec.jacobians[i];
      if (j.pattern == sparsity::dense) {
        s << "  moto_linear_cond_grad_dense(j" << i << ',' << j.rows << ','
          << j.cols << ",residual," << j.row_offset << ",g" << i << ','
          << j.col_offset << ");\n";
      } else if (j.pattern == sparsity::diag)
        s << "  moto_linear_cond_grad_diag(j" << i << ',' << j.rows
          << ",residual," << j.row_offset << ",g" << i << ',' << j.col_offset
          << ");\n";
      else
        s << "  moto_linear_cond_grad_eye(" << j.rows << ",residual,"
          << j.row_offset << ",g" << i << ',' << j.col_offset << ");\n";
    }
    for (const auto [i, j] : pairs) {
      const auto &lhs = spec.jacobians[i];
      const auto &rhs = spec.jacobians[j];
      const size_t row = std::max(lhs.row_offset, rhs.row_offset);
      const size_t end =
          std::min(lhs.row_offset + lhs.rows, rhs.row_offset + rhs.rows);
      if (end <= row)
        continue;
      const size_t n = end - row;
      const size_t li = row - lhs.row_offset;
      const size_t ri = row - rhs.row_offset;
      const auto pname = [](sparsity p) {
        return p == sparsity::dense  ? "dense"
               : p == sparsity::diag ? "diag"
                                     : "eye";
      };
      s << "  moto_linear_cond_hess_" << pname(lhs.pattern) << '_'
        << pname(rhs.pattern) << "(j" << i << ',' << lhs.rows << ',' << lhs.cols
        << ',' << li << ",j" << j << ',' << rhs.rows << ',' << rhs.cols << ','
        << ri << ",weight," << row << ',' << n << ",h" << i << '_' << j
        << ");\n";
    }
    s << "}\n";
    return s.str();
  }

  for (size_t i = 0; i < nargs; ++i) {
    s << "  moto_linear_cond_grad_dense(j" << i << ',' << spec.rows << ','
      << spec.jac_cols[i] << ",residual,0,g" << i << ",0);\n";
  }
  for (size_t i = 0; i < nargs; ++i) {
    for (size_t j = i; j < nargs; ++j) {
      s << "  moto_linear_cond_hess_dense_dense(j" << i << ',' << spec.rows
        << ',' << spec.jac_cols[i] << ",0,j" << j << ',' << spec.rows << ','
        << spec.jac_cols[j] << ",0,weight,0," << spec.rows << ",h" << i << '_'
        << j << ");\n";
    }
  }
  s << "}\n";
  return s.str();
}

std::string emit_condensation_source(const condensation_spec &spec) {
  return emit_condensation_function(spec, symbol_name, true);
}

condensation_kernel
compile_condensation(condensation_spec spec,
                     const std::filesystem::path &cache_dir) {
  const std::string source = emit_condensation_source(spec);
  const std::string key = utils::compute_md5_from_bytes(source);
  auto function = reinterpret_cast<condensation_kernel::function_type>(
      compile_source(source, cache_dir));
  return condensation_kernel(std::move(spec), function, key);
}

batch_condensation_kernel
compile_batch_condensation(batch_condensation_spec spec,
                           const std::filesystem::path &cache_dir) {
  if (spec.constraints.empty())
    throw std::invalid_argument("empty OCP condensation batch");
  std::ostringstream signature;
  for (const auto &constraint : spec.constraints) {
    signature << constraint.rows << ':';
    for (const auto &p : constraint.jacobians)
      signature << static_cast<int>(p.pattern) << ',' << p.row_offset << ','
                << p.col_offset << ',' << p.rows << ',' << p.cols << ';';
    signature << '/';
    for (const auto sign : constraint.residual_signs)
      signature << sign << ',';
    signature << '/';
    for (const auto [i, j] : constraint.hessian_pairs)
      signature << i << ',' << j << ';';
    signature << '|';
  }
  const auto key = signature.str();
  {
    std::lock_guard lock(batch_mutex);
    if (auto found = condensation_kernels.find(key);
        found != condensation_kernels.end())
      return found->second;
  }
  std::ostringstream source;
  std::vector<size_t> offsets;
  size_t offset = 0;
  for (size_t i = 0; i < spec.constraints.size(); ++i) {
    spec.constraints[i].validate();
    offsets.push_back(offset);
    condensation_kernel layout(spec.constraints[i], nullptr, {});
    offset += layout.pointer_count();
    source << emit_condensation_function(
        spec.constraints[i], "moto_linear_jit_cond_" + std::to_string(i),
        false);
  }
  source << "extern \"C\" __attribute__((visibility(\"default\"))) void "
         << symbol_name << "(double *const *p) {\n";
  for (size_t i = 0; i < offsets.size(); ++i)
    source << "  moto_linear_jit_cond_" << i << "(p+" << offsets[i] << ");\n";
  source << "}\n";
  auto function = reinterpret_cast<batch_condensation_kernel::function_type>(
      compile_source(source.str(), cache_dir));
  auto kernel = batch_condensation_kernel(offset, function);
  std::lock_guard lock(batch_mutex);
  return condensation_kernels.emplace(key, kernel).first->second;
}

} // namespace moto::linear_backend
