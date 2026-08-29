#include <moto/core/linear_backend.hpp>

#include "casadi_mx_graph_translator.hpp"

#include <moto/core/external_function.hpp>
#include <moto/core/sparse_matrix.hpp>
#include <moto/utils/codegen.hpp>

#include <Eigen/Cholesky>
#include <Eigen/LU>

#include <algorithm>
#include <bit>
#include <dlfcn.h>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <unistd.h>

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
std::mutex graph_mutex;
std::unordered_map<std::string,
                   std::weak_ptr<const detail::casadi_mx_graph_plan>>
    graph_plans;
std::unordered_map<std::string, std::shared_ptr<std::mutex>> graph_compile_locks;
void collect_mx_solves(
    const casadi::MX &expression,
    std::unordered_set<const casadi::MXNode *> &visited,
    std::vector<casadi::MX> &solves) {
  if (!expression.get() || !visited.insert(expression.get()).second) return;
  if (expression.op() == casadi::OP_SOLVE) solves.push_back(expression);
  for (casadi_int i = 0; i < expression.n_dep(); ++i)
    collect_mx_solves(expression.dep(i), visited, solves);
}

void collect_mx_products(
    const casadi::MX &expression,
    std::unordered_set<const casadi::MXNode *> &visited,
    std::vector<casadi::MX> &products) {
  if (!expression.get() || !visited.insert(expression.get()).second) return;
  if (expression.op() == casadi::OP_MTIMES && expression.n_dep() == 3 &&
      expression.dep(0).is_zero())
    products.push_back(expression);
  for (casadi_int i = 0; i < expression.n_dep(); ++i)
    collect_mx_products(expression.dep(i), visited, products);
}

using mx_parent_map =
    std::unordered_map<const casadi::MXNode *, std::vector<casadi::MX>>;

void collect_mx_parents(const casadi::MX &expression,
                        std::unordered_set<const casadi::MXNode *> &visited,
                        mx_parent_map &parents) {
  if (!expression.get() || !visited.insert(expression.get()).second) return;
  for (casadi_int i = 0; i < expression.n_dep(); ++i) {
    const casadi::MX dependency = expression.dep(i);
    parents[dependency.get()].push_back(expression);
    collect_mx_parents(dependency, visited, parents);
  }
}

bool mx_contains_node(const casadi::MX &expression,
                      const casadi::MXNode *target,
                      std::unordered_set<const casadi::MXNode *> &visited) {
  if (!expression.get()) return false;
  if (expression.get() == target) return true;
  if (!visited.insert(expression.get()).second) return false;
  for (casadi_int i = 0; i < expression.n_dep(); ++i)
    if (mx_contains_node(expression.dep(i), target, visited)) return true;
  return false;
}

std::string mx_structural_key(
    const casadi::MX &expression,
    std::unordered_map<const casadi::MXNode *, std::string> &cache) {
  if (const auto found = cache.find(expression.get()); found != cache.end())
    return found->second;
  std::ostringstream description;
  description << expression.op() << ':' << expression.size1() << ':'
              << expression.size2() << ':';
  if (expression.is_symbolic()) description << expression.name();
  if (expression.is_constant())
    for (const double value : casadi::MX::evalf(expression).nonzeros())
      description << value << ',';
  for (const auto &[name, value] : expression.info())
    description << name << '=' << value << ';';
  const auto &sparsity = expression.sparsity();
  for (const casadi_int value : sparsity.get_colind())
    description << value << ',';
  description << ':';
  for (const casadi_int value : sparsity.get_row())
    description << value << ',';
  for (casadi_int i = 0; i < expression.n_dep(); ++i)
    description << ':' << mx_structural_key(expression.dep(i), cache);
  const std::string key =
      utils::compute_md5_from_bytes(description.str());
  cache.emplace(expression.get(), key);
  return key;
}

std::vector<std::vector<casadi::MX>> batch_independent_mx_solves(
    std::vector<std::vector<casadi::MX>> output_entries) {
  struct solve_record {
    casadi::MX representative;
    std::vector<casadi::MX> nodes;
    uint64_t usage = 0;
  };
  std::set<std::string> batched_matrix_groups;
  std::unordered_map<const casadi::MXNode *, std::string> structural_keys;
  for (;;) {
    std::vector<solve_record> records;
    std::unordered_map<std::string, size_t> record_by_key;
    for (size_t entry = 0; entry < output_entries.size(); ++entry) {
      std::unordered_set<const casadi::MXNode *> visited;
      std::vector<casadi::MX> entry_solves;
      for (const auto &output : output_entries[entry])
        collect_mx_solves(output, visited, entry_solves);
      for (const auto &solve : entry_solves) {
        const std::string expression_key =
            mx_structural_key(solve, structural_keys);
        const auto [found, inserted] =
            record_by_key.try_emplace(expression_key, records.size());
        if (inserted) records.push_back({solve, {}, 0});
        auto &record = records[found->second];
        record.nodes.push_back(solve);
        record.usage |= uint64_t{1} << entry;
      }
    }
    struct solve_group {
      std::string key;
      std::vector<solve_record *> records;
    };
    std::vector<solve_group> groups;
    std::unordered_map<std::string, size_t> group_by_key;
    for (auto &record : records) {
      const auto &solve = record.representative;
      const auto info = solve.info();
      const bool transpose =
          info.contains("tr") && info.at("tr").to_bool();
      if (transpose) continue;
      const std::string matrix_key =
          mx_structural_key(solve.dep(1), structural_keys);
      const std::string group_key =
          std::to_string(record.usage) + ':' + matrix_key;
      if (batched_matrix_groups.contains(group_key)) continue;
      const auto [found, inserted] =
          group_by_key.try_emplace(group_key, groups.size());
      if (inserted) groups.push_back({group_key, {}});
      groups[found->second].records.push_back(&record);
    }

    bool updated = false;
    for (auto group_it = groups.rbegin(); group_it != groups.rend();
         ++group_it) {
      auto &[key, group] = *group_it;
      if (group.size() < 2) continue;
      bool independent = true;
      for (size_t i = 0; i < group.size() && independent; ++i)
        for (size_t j = 0; j < group.size(); ++j) {
          if (i == j) continue;
          std::unordered_set<const casadi::MXNode *> dependency_visited;
          const auto &lhs = group[i]->representative;
          const auto &rhs = group[j]->representative;
          if (mx_contains_node(lhs.dep(0), rhs.get(),
                               dependency_visited) ||
              mx_contains_node(lhs.dep(1), rhs.get(),
                               dependency_visited)) {
            independent = false;
            break;
          }
        }
      if (!independent) continue;

      std::vector<casadi::MX> right_hand_sides;
      right_hand_sides.reserve(group.size());
      for (const auto *record : group)
        right_hand_sides.push_back(record->representative.dep(0));
      const casadi::MX batched = casadi::MX::solve(
          group.front()->representative.dep(1),
          casadi::MX::horzcat(right_hand_sides));
      std::vector<casadi::MX> targets, replacements;
      casadi_int column = 0;
      for (const auto *record : group) {
        const auto &solve = record->representative;
        const casadi::MX replacement = batched(
            casadi::Slice(),
            casadi::Slice(column, column + solve.size2()));
        for (const auto &node : record->nodes) {
          targets.push_back(node);
          replacements.push_back(replacement);
        }
        column += solve.size2();
      }
      for (auto &outputs : output_entries)
        outputs =
            casadi::MX::graph_substitute(outputs, targets, replacements);
      batched_matrix_groups.insert(key);
      updated = true;
      break;
    }
    if (!updated) return output_entries;
  }
}

std::vector<std::vector<casadi::MX>> batch_independent_mx_products(
    std::vector<std::vector<casadi::MX>> output_entries) {
  struct product_record {
    casadi::MX representative;
    std::vector<casadi::MX> nodes;
    uint64_t usage = 0;
  };
  std::set<std::string> batched_left_groups;
  std::unordered_map<const casadi::MXNode *, std::string> structural_keys;
  for (;;) {
    mx_parent_map parents;
    {
      std::unordered_set<const casadi::MXNode *> visited;
      for (const auto &outputs : output_entries)
        for (const auto &output : outputs)
          collect_mx_parents(output, visited, parents);
    }
    std::vector<product_record> records;
    std::unordered_map<std::string, size_t> record_by_key;
    for (size_t entry = 0; entry < output_entries.size(); ++entry) {
      std::unordered_set<const casadi::MXNode *> visited;
      std::vector<casadi::MX> entry_products;
      for (const auto &output : output_entries[entry])
        collect_mx_products(output, visited, entry_products);
      for (const auto &product : entry_products) {
        const std::string expression_key =
            mx_structural_key(product, structural_keys);
        const auto [found, inserted] =
            record_by_key.try_emplace(expression_key, records.size());
        if (inserted) records.push_back({product, {}, 0});
        auto &record = records[found->second];
        record.nodes.push_back(product);
        record.usage |= uint64_t{1} << entry;
      }
    }

    struct product_group {
      std::string key;
      std::vector<product_record *> records;
    };
    std::vector<product_group> groups;
    std::unordered_map<std::string, size_t> group_by_key;
    for (auto &record : records) {
      const auto &product = record.representative;
      const std::string left_key =
          mx_structural_key(product.dep(1), structural_keys);
      const std::string group_key =
          std::to_string(record.usage) + ':' + left_key;
      if (batched_left_groups.contains(group_key)) continue;
      const auto [found, inserted] =
          group_by_key.try_emplace(group_key, groups.size());
      if (inserted) groups.push_back({group_key, {}});
      groups[found->second].records.push_back(&record);
    }

    bool updated = false;
    for (auto group_it = groups.rbegin(); group_it != groups.rend();
         ++group_it) {
      auto &[key, group] = *group_it;
      if (group.size() < 2) continue;
      bool independent = true;
      for (size_t i = 0; i < group.size() && independent; ++i)
        for (size_t j = 0; j < group.size(); ++j) {
          if (i == j) continue;
          std::unordered_set<const casadi::MXNode *> dependency_visited;
          if (mx_contains_node(group[i]->representative.dep(2),
                               group[j]->representative.get(),
                               dependency_visited)) {
            independent = false;
            break;
          }
        }
      if (!independent) continue;

      std::vector<casadi::MX> right_operands;
      right_operands.reserve(group.size());
      struct fma_record {
        casadi::MX addend;
        double product_sign = 1.;
        std::vector<casadi::MX> parents;
      };
      std::vector<fma_record> fma;
      fma.reserve(group.size());
      bool all_fma = true;
      for (const auto *record : group) {
        fma_record item;
        bool initialized = false;
        for (const auto &node : record->nodes) {
          const auto found = parents.find(node.get());
          if (found == parents.end() || found->second.size() != 1) {
            all_fma = false;
            break;
          }
          const casadi::MX parent = found->second.front();
          if (parent.op() != casadi::OP_ADD &&
              parent.op() != casadi::OP_SUB) {
            all_fma = false;
            break;
          }
          const bool product_is_lhs = parent.dep(0).get() == node.get();
          const bool product_is_rhs = parent.dep(1).get() == node.get();
          if (!product_is_lhs && !product_is_rhs) {
            all_fma = false;
            break;
          }
          casadi::MX addend = product_is_lhs ? parent.dep(1) : parent.dep(0);
          double addend_sign = 1., product_sign = 1.;
          if (parent.op() == casadi::OP_SUB) {
            if (product_is_lhs)
              addend_sign = -1.;
            else
              product_sign = -1.;
          }
          if (addend_sign < 0.) addend = -addend;
          if (!initialized) {
            item.addend = addend;
            item.product_sign = product_sign;
            initialized = true;
          } else if (item.product_sign != product_sign ||
                     mx_structural_key(item.addend, structural_keys) !=
                         mx_structural_key(addend, structural_keys)) {
            all_fma = false;
            break;
          }
          item.parents.push_back(parent);
        }
        if (!all_fma) break;
        fma.push_back(std::move(item));
      }
      for (size_t i = 0; i < group.size(); ++i) {
        casadi::MX right = group[i]->representative.dep(2);
        if (all_fma && fma[i].product_sign < 0.) right = -right;
        right_operands.push_back(std::move(right));
      }
      const casadi::MX batched_rhs = casadi::MX::horzcat(right_operands);
      const casadi::MX batched =
          all_fma
              ? casadi::MX::mac(group.front()->representative.dep(1),
                                batched_rhs,
                                casadi::MX::horzcat([&] {
                                  std::vector<casadi::MX> addends;
                                  addends.reserve(fma.size());
                                  for (const auto &item : fma)
                                    addends.push_back(item.addend);
                                  return addends;
                                }()))
              : casadi::MX::mtimes(group.front()->representative.dep(1),
                                    batched_rhs);
      std::vector<casadi::MX> targets, replacements;
      casadi_int column = 0;
      for (size_t i = 0; i < group.size(); ++i) {
        const auto *record = group[i];
        const auto &product = record->representative;
        const casadi::MX replacement = batched(
            casadi::Slice(),
            casadi::Slice(column, column + product.size2()));
        const auto &nodes = all_fma ? fma[i].parents : record->nodes;
        for (const auto &node : nodes) {
          targets.push_back(node);
          replacements.push_back(replacement);
        }
        column += product.size2();
      }
      for (auto &outputs : output_entries)
        outputs =
            casadi::MX::graph_substitute(outputs, targets, replacements);
      batched_left_groups.insert(key);
      updated = true;
      break;
    }
    if (!updated) return output_entries;
  }
}

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
                     const std::filesystem::path &cache_dir,
                     std::string_view optimization = "-O3") {
  const std::string key = utils::compute_md5_from_bytes(
      optimization == "-O3"
          ? source
          : source + "\n:moto-jit-optimization:" +
                std::string(optimization));
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
          "g++ -shared -fPIC -std=c++20 " + std::string(optimization) +
          " -DNDEBUG -march=native "
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

void *detail::compile_casadi_mx_graph_source(
    const std::string &source, const std::filesystem::path &cache_dir) {
  return compile_source(
      source, cache_dir,
      "-O1 -ftree-loop-vectorize -fvect-cost-model=unlimited "
      "-fstrict-aliasing");
}

namespace {
struct graph_factor_state {
  struct factor {
    Eigen::PartialPivLU<matrix> lu;
    Eigen::LLT<matrix> llt;
    bool spd = false;

    void compute(const Eigen::Ref<const matrix> &value, bool use_spd) {
      spd = use_spd;
      if (spd) {
        llt.compute(value);
        if (llt.info() != Eigen::Success)
          throw std::runtime_error(
              "declared SPD graph factor is not positive definite");
      } else {
        lu.compute(value);
      }
    }
    matrix inverse() const {
      if (spd)
        return llt.solve(matrix::Identity(llt.rows(), llt.cols()));
      return lu.inverse();
    }
  };
  size_t epoch = 1;
  std::vector<size_t> factor_epoch;
  std::vector<factor> factors;
  explicit graph_factor_state(size_t count)
      : factor_epoch(count), factors(count) {}
};

} // namespace

namespace {
template <int N>
void graph_small_inverse(const double *a, double *output) {
  using small_matrix = Eigen::Matrix<double, N, N>;
  const Eigen::Map<const small_matrix> value(a);
  Eigen::Map<small_matrix> inverse(output);
  inverse = value.partialPivLu().inverse();
}

template <int N>
void graph_small_solve(const double *a, bool transpose, const double *rhs,
                       size_t cols, size_t rhs_leading, double *output,
                       size_t output_leading) {
  using small_matrix = Eigen::Matrix<double, N, N>;
  using rhs_matrix = Eigen::Matrix<double, N, Eigen::Dynamic>;
  const Eigen::PartialPivLU<small_matrix> factor{
      Eigen::Map<const small_matrix>(a)};
  const Eigen::Map<const rhs_matrix, Eigen::Unaligned,
                   Eigen::OuterStride<>> b(
      rhs, N, cols, Eigen::OuterStride<>(rhs_leading));
  Eigen::Map<rhs_matrix, Eigen::Unaligned, Eigen::OuterStride<>> x(
      output, N, cols, Eigen::OuterStride<>(output_leading));
  if (transpose) x = factor.transpose().solve(b);
  else x = factor.solve(b);
}
} // namespace

extern "C" __attribute__((visibility("default"))) void
moto_graph_small_inverse(const double *a, double *output, size_t n) {
  if (n == 1) {
    output[0] = 1. / a[0];
    return;
  }
  if (n == 2) {
    graph_small_inverse<2>(a, output);
    return;
  }
  graph_small_inverse<3>(a, output);
}

extern "C" __attribute__((visibility("default"))) void
moto_graph_small_solve(const double *a, bool transpose, const double *rhs,
                       size_t n, size_t cols, size_t rhs_leading,
                       double *output, size_t output_leading) {
  if (n == 1) {
    for (size_t col = 0; col < cols; ++col)
      output[col * output_leading] = rhs[col * rhs_leading] / a[0];
  } else if (n == 2) {
    graph_small_solve<2>(a, transpose, rhs, cols, rhs_leading, output,
                         output_leading);
  } else {
    graph_small_solve<3>(a, transpose, rhs, cols, rhs_leading, output,
                         output_leading);
  }
}

struct moto_graph_copy_run {
  size_t pointer, panel_offset, count;
  ptrdiff_t panel_stride, local_offset, local_stride;
};

extern "C" __attribute__((visibility("default"))) void moto_graph_pack(
    double *local, double **p, const moto_graph_copy_run *runs,
    size_t run_count) {
  for (size_t run_index = 0; run_index < run_count; ++run_index) {
    const auto &run = runs[run_index];
    for (size_t i = 0; i < run.count; ++i) {
      const auto ordinal = static_cast<ptrdiff_t>(i);
      local[run.local_offset + run.local_stride * ordinal] =
          p[run.pointer][static_cast<ptrdiff_t>(run.panel_offset) +
                         run.panel_stride * ordinal];
    }
  }
}

extern "C" __attribute__((visibility("default"))) void moto_graph_unpack(
    double **p, const moto_graph_copy_run *runs, size_t run_count,
    const double *local) {
  for (size_t run_index = 0; run_index < run_count; ++run_index) {
    const auto &run = runs[run_index];
    double *destination = p[run.pointer] + run.panel_offset;
    if (run.local_offset < 0) {
      for (size_t i = 0; i < run.count; ++i) {
        const auto ordinal = static_cast<ptrdiff_t>(i);
        destination[run.panel_stride * ordinal] = 0.;
      }
    } else {
      for (size_t i = 0; i < run.count; ++i) {
        const auto ordinal = static_cast<ptrdiff_t>(i);
        destination[run.panel_stride * ordinal] =
            local[run.local_offset + run.local_stride * ordinal];
      }
    }
  }
}

extern "C" __attribute__((visibility("default"))) void *
moto_graph_factor_state_create(size_t factors) {
  return new graph_factor_state(factors);
}

extern "C" __attribute__((visibility("default"))) void
moto_graph_factor_state_destroy(void *opaque) {
  delete static_cast<graph_factor_state *>(opaque);
}

extern "C" __attribute__((visibility("default"))) void
moto_graph_factor_next_epoch(void *opaque) {
  ++static_cast<graph_factor_state *>(opaque)->epoch;
}

extern "C" __attribute__((visibility("default"))) void moto_graph_factor(
    void *opaque, size_t slot, bool spd, const double *a, size_t n) {
  auto &state = *static_cast<graph_factor_state *>(opaque);
  if (state.factor_epoch[slot] == state.epoch) return;
  const Eigen::Map<const matrix> value(a, n, n);
  state.factors[slot].compute(value, spd);
  state.factor_epoch[slot] = state.epoch;
}

extern "C" __attribute__((visibility("default"))) void moto_graph_inverse(
    void *opaque, size_t slot, double *output, size_t n) {
  auto &factor = static_cast<graph_factor_state *>(opaque)->factors[slot];
  Eigen::Map<matrix> inverse(output, n, n);
  inverse = factor.inverse();
}

extern "C" __attribute__((visibility("default"))) void moto_graph_solve(
    void *opaque, size_t slot, bool transpose, const double *rhs, size_t n,
    size_t cols, size_t rhs_leading, double *output,
    size_t output_leading) {
  auto &factor = static_cast<graph_factor_state *>(opaque)->factors[slot];
  Eigen::Map<const matrix, Eigen::Unaligned, Eigen::OuterStride<>> b(
      rhs, n, cols, Eigen::OuterStride<>(rhs_leading));
  Eigen::Map<matrix, Eigen::Unaligned, Eigen::OuterStride<>> x(
      output, n, cols, Eigen::OuterStride<>(output_leading));
  if (factor.spd) x = factor.llt.solve(b);
  else if (transpose) x = factor.lu.transpose().solve(b);
  else x = factor.lu.solve(b);
}

template <int Alignment = Eigen::Unaligned>
using const_strided_matrix_map =
    Eigen::Map<const matrix, Alignment, Eigen::OuterStride<>>;
template <int Alignment = Eigen::Unaligned>
using strided_matrix_map = Eigen::Map<matrix, Alignment, Eigen::OuterStride<>>;

template <int Alignment = Eigen::Unaligned>
auto const_matrix_view(const double *data, size_t rows, size_t cols,
                       size_t outer_stride) {
  return const_strided_matrix_map<Alignment>(
      data, rows, cols, Eigen::OuterStride<>(outer_stride));
}

template <int Alignment = Eigen::Unaligned>
auto matrix_view(double *data, size_t rows, size_t cols, size_t outer_stride) {
  return strided_matrix_map<Alignment>(data, rows, cols,
                                       Eigen::OuterStride<>(outer_stride));
}

extern "C" __attribute__((visibility("default"))) void
moto_linear_dense_times(const double *a, size_t ar, size_t ac, const double *b,
                        size_t br, size_t bc, double *o, size_t orows,
                        size_t ocols, size_t ro, size_t co, double alpha) {
  Eigen::Map<const matrix, Eigen::Aligned> A(a, ar, ac);
  const auto B = const_matrix_view(b + co, ac, bc, br);
  auto O = matrix_view(o + ro, ar, bc, orows);
  O.noalias() += alpha * A * B;
}

extern "C" __attribute__((visibility("default"))) void
moto_linear_dense_transpose_times(const double *a, size_t ar, size_t ac,
                                  const double *b, size_t br, size_t bc,
                                  double *o, size_t orows, size_t ocols,
                                  size_t ro, size_t co, double alpha) {
  Eigen::Map<const matrix, Eigen::Aligned> A(a, ar, ac);
  const auto B = const_matrix_view(b + ro, ar, bc, br);
  auto O = matrix_view(o + co, ac, bc, orows);
  O.noalias() += alpha * A.transpose() * B;
}

extern "C" __attribute__((visibility("default"))) void
moto_linear_dense_right_times(const double *a, size_t ar, size_t ac,
                              const double *b, size_t br, size_t bc, double *o,
                              size_t orows, size_t ocols, size_t ro, size_t co,
                              double alpha) {
  Eigen::Map<const matrix, Eigen::Aligned> A(a, ar, ac);
  const auto B = const_matrix_view(b + ro * br, br, ar, br);
  auto O = matrix_view(o + co * orows, br, ac, orows);
  O.noalias() += alpha * B * A;
}

extern "C" __attribute__((visibility("default"))) void
moto_linear_dense_right_transpose_times(const double *a, size_t ar, size_t ac,
                                        const double *b, size_t br, size_t bc,
                                        double *o, size_t orows, size_t ocols,
                                        size_t ro, size_t co, double alpha) {
  Eigen::Map<const matrix, Eigen::Aligned> A(a, ar, ac);
  const auto B = const_matrix_view(b + ro, ar, bc, br);
  auto O = matrix_view(o + co * orows, bc, ac, orows);
  O.noalias() += alpha * B.transpose() * A;
}

template <product_op Op, bool Eye>
void structured_product(const double *a, size_t n, const double *b, size_t br,
                        size_t bc, double *o, size_t orows, size_t ocols,
                        size_t ro, size_t co, double alpha) {
  if constexpr (Op == product_op::times) {
    const auto B = const_matrix_view(b + co, n, bc, br);
    auto O = matrix_view(o + ro, n, bc, orows);
    if constexpr (Eye)
      O += alpha * B;
    else
      O.noalias() +=
          alpha * Eigen::Map<const vector, Eigen::Aligned>(a, n).asDiagonal() *
          B;
  } else if constexpr (Op == product_op::transpose_times) {
    const auto B = const_matrix_view(b + ro, n, bc, br);
    auto O = matrix_view(o + co, n, bc, orows);
    if constexpr (Eye)
      O += alpha * B;
    else
      O.noalias() +=
          alpha * Eigen::Map<const vector, Eigen::Aligned>(a, n).asDiagonal() *
          B;
  } else if constexpr (Op == product_op::right_times) {
    const auto B = const_matrix_view(b + ro * br, br, n, br);
    auto O = matrix_view(o + co * orows, br, n, orows);
    if constexpr (Eye)
      O += alpha * B;
    else
      O.noalias() += alpha * B *
          Eigen::Map<const vector, Eigen::Aligned>(a, n).asDiagonal();
  } else {
    const auto B = const_matrix_view(b + ro, n, bc, br);
    auto O = matrix_view(o + co * orows, bc, n, orows);
    if constexpr (Eye)
      O += alpha * B.transpose();
    else
      O.noalias() += alpha * B.transpose() *
          Eigen::Map<const vector, Eigen::Aligned>(a, n).asDiagonal();
  }
}

template <product_op Op, bool Dense>
void fused_pair_product(const double *a, const double *a1, size_t ar,
                        size_t ac, const double *b, size_t br, size_t bc,
                        double *o, size_t orows, size_t ocols, size_t ro,
                        size_t co, double alpha) {
  if constexpr (Dense) {
    const Eigen::Map<const matrix, Eigen::Aligned> A(a, ar, ac), A1(a1, ar, ac);
    if constexpr (Op == product_op::times) {
      const auto B = const_matrix_view(b + co, ac, bc, br);
      auto O = matrix_view(o + ro, ar, bc, orows);
      O.noalias() += alpha * (A + A1) * B;
    } else if constexpr (Op == product_op::transpose_times) {
      const auto B = const_matrix_view(b + ro, ar, bc, br);
      auto O = matrix_view(o + co, ac, bc, orows);
      O.noalias() += alpha * (A + A1).transpose() * B;
    } else if constexpr (Op == product_op::right_times) {
      const auto B = const_matrix_view(b + ro * br, br, ar, br);
      auto O = matrix_view(o + co * orows, br, ac, orows);
      O.noalias() += alpha * B * (A + A1);
    } else {
      const auto B = const_matrix_view(b + ro, ar, bc, br);
      auto O = matrix_view(o + co * orows, bc, ac, orows);
      O.noalias() += alpha * B.transpose() * (A + A1);
    }
  } else {
    const auto d = Eigen::Map<const vector, Eigen::Aligned>(a, ar).array() +
                   Eigen::Map<const vector, Eigen::Aligned>(a1, ar).array();
    if constexpr (Op == product_op::times) {
      const auto B = const_matrix_view(b + co, ar, bc, br);
      auto O = matrix_view(o + ro, ar, bc, orows);
      O.array() += alpha * (B.array().colwise() * d);
    } else if constexpr (Op == product_op::transpose_times) {
      const auto B = const_matrix_view(b + ro, ar, bc, br);
      auto O = matrix_view(o + co, ar, bc, orows);
      O.array() += alpha * (B.array().colwise() * d);
    } else if constexpr (Op == product_op::right_times) {
      const auto B = const_matrix_view(b + ro * br, br, ar, br);
      auto O = matrix_view(o + co * orows, br, ar, orows);
      O.array() += alpha * (B.array().rowwise() * d.transpose());
    } else {
      const auto B = const_matrix_view(b + ro, ar, bc, br);
      auto O = matrix_view(o + co * orows, bc, ar, orows);
      O.array() += alpha * (B.transpose().array().rowwise() * d.transpose());
    }
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

extern "C" __attribute__((visibility("default"))) void
moto_linear_row_scale_dense(double *a, size_t rows, size_t cols,
                            const double *scale, size_t row_offset) {
  Eigen::Map<matrix, Eigen::Aligned>(a, rows, cols).array().colwise() *=
      Eigen::Map<const vector>(scale + row_offset, rows).array();
}

extern "C" __attribute__((visibility("default"))) void
moto_linear_row_scale_diag(double *a, size_t rows, const double *scale,
                           size_t row_offset) {
  Eigen::Map<vector, Eigen::Aligned>(a, rows).array() *=
      Eigen::Map<const vector>(scale + row_offset, rows).array();
}

template <bool Scaled>
void row_infnorm_dense(const double *a, size_t rows, size_t cols,
                       const double *scale, double *norms, size_t row_offset) {
  const auto values = Eigen::Map<const matrix, Eigen::Aligned>(a, rows, cols)
                          .cwiseAbs()
                          .rowwise()
                          .maxCoeff();
  auto output = Eigen::Map<vector>(norms + row_offset, rows);
  if constexpr (Scaled)
    output.array() = output.array().max(
        values.array() *
        Eigen::Map<const vector>(scale + row_offset, rows).cwiseAbs().array());
  else
    output = output.cwiseMax(values);
}

template <bool Scaled>
void row_infnorm_diag(const double *a, size_t rows, const double *scale,
                      double *norms, size_t row_offset) {
  auto values = Eigen::Map<const vector, Eigen::Aligned>(a, rows).cwiseAbs();
  auto output = Eigen::Map<vector>(norms + row_offset, rows);
  if constexpr (Scaled)
    output = output.cwiseMax(
        values.cwiseProduct(Eigen::Map<const vector>(scale + row_offset, rows)
                                .cwiseAbs()));
  else
    output = output.cwiseMax(values);
}

extern "C" __attribute__((visibility("default"))) void
moto_linear_row_infnorm_dense(const double *a, size_t rows, size_t cols,
                              const double *, double *norms,
                              size_t row_offset) {
  row_infnorm_dense<false>(a, rows, cols, nullptr, norms, row_offset);
}
extern "C" __attribute__((visibility("default"))) void
moto_linear_scaled_row_infnorm_dense(const double *a, size_t rows, size_t cols,
                                     const double *scale, double *norms,
                                     size_t row_offset) {
  row_infnorm_dense<true>(a, rows, cols, scale, norms, row_offset);
}
extern "C" __attribute__((visibility("default"))) void
moto_linear_row_infnorm_diag(const double *a, size_t rows, size_t,
                             const double *, double *norms,
                             size_t row_offset) {
  row_infnorm_diag<false>(a, rows, nullptr, norms, row_offset);
}
extern "C" __attribute__((visibility("default"))) void
moto_linear_scaled_row_infnorm_diag(const double *a, size_t rows, size_t,
                                    const double *scale, double *norms,
                                    size_t row_offset) {
  row_infnorm_diag<true>(a, rows, scale, norms, row_offset);
}
extern "C" __attribute__((visibility("default"))) void
moto_linear_row_infnorm_eye(const double *, size_t rows, size_t,
                            const double *, double *norms, size_t row_offset) {
  auto output = Eigen::Map<vector>(norms + row_offset, rows);
  output.array() = output.array().max(1.);
}
extern "C" __attribute__((visibility("default"))) void
moto_linear_scaled_row_infnorm_eye(const double *, size_t rows, size_t,
                                   const double *scale, double *norms,
                                   size_t row_offset) {
  auto output = Eigen::Map<vector>(norms + row_offset, rows);
  output = output.cwiseMax(
      Eigen::Map<const vector>(scale + row_offset, rows).cwiseAbs());
}

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

template <bool LhsTranspose, bool RhsTranspose,
          int LhsAlignment = Eigen::Unaligned,
          int RhsAlignment = Eigen::Unaligned>
void pair_dense_dense(const double *a, size_t ar, size_t ac, size_t ald,
                      size_t ak, const double *b, size_t br, size_t bc,
                      size_t bld, size_t bk,
                      double *out, size_t out_rows, size_t out_row,
                      size_t out_col, size_t n, double alpha) {
  const auto A = [&] {
    if constexpr (LhsTranspose)
      return const_matrix_view<LhsAlignment>(a + ak, n, ac, ald);
    else
      return const_matrix_view<LhsAlignment>(a + ak * ald, ar, n, ald);
  }();
  const auto B = [&] {
    if constexpr (RhsTranspose)
      return const_matrix_view<RhsAlignment>(b + bk * bld, br, n, bld);
    else
      return const_matrix_view<RhsAlignment>(b + bk, n, bc, bld);
  }();
  auto O = matrix_view(out + out_row + out_col * out_rows,
                       LhsTranspose ? ac : ar,
                       RhsTranspose ? br : bc, out_rows);
  if constexpr (LhsTranspose && RhsTranspose)
    O.noalias() += alpha * A.transpose() * B.transpose();
  else if constexpr (LhsTranspose)
    O.noalias() += alpha * A.transpose() * B;
  else if constexpr (RhsTranspose)
    O.noalias() += alpha * A * B.transpose();
  else
    O.noalias() += alpha * A * B;
}

template <bool Eye, bool RhsTranspose, int LhsAlignment = Eigen::Unaligned,
          int RhsAlignment = Eigen::Unaligned>
void pair_struct_dense(const double *a, size_t ak, const double *b, size_t br,
                       size_t bc, size_t bld, size_t bk, double *out,
                       size_t out_rows,
                       size_t out_row, size_t out_col, size_t n, double alpha) {
  const auto B = [&] {
    if constexpr (RhsTranspose)
      return const_matrix_view<RhsAlignment>(b + bk * bld, br, n, bld);
    else
      return const_matrix_view<RhsAlignment>(b + bk, n, bc, bld);
  }();
  auto O = matrix_view(out + out_row + out_col * out_rows, n,
                       RhsTranspose ? br : bc, out_rows);
  if constexpr (RhsTranspose) {
    if constexpr (Eye)
      O += alpha * B.transpose();
    else
      O.noalias() += alpha *
                     Eigen::Map<const vector, LhsAlignment>(a + ak, n)
                         .asDiagonal() *
                     B.transpose();
  } else if constexpr (Eye) {
    O += alpha * B;
  } else {
    O.noalias() += alpha *
                   Eigen::Map<const vector, LhsAlignment>(a + ak, n)
                       .asDiagonal() *
                   B;
  }
}

template <bool LhsTranspose, bool Eye, int LhsAlignment = Eigen::Unaligned,
          int RhsAlignment = Eigen::Unaligned>
void pair_dense_struct(const double *a, size_t ar, size_t ac, size_t ald,
                       size_t ak,
                       const double *b, size_t bk, double *out, size_t out_rows,
                       size_t out_row, size_t out_col, size_t n, double alpha) {
  const auto A = [&] {
    if constexpr (LhsTranspose)
      return const_matrix_view<LhsAlignment>(a + ak, n, ac, ald);
    else
      return const_matrix_view<LhsAlignment>(a + ak * ald, ar, n, ald);
  }();
  auto O = matrix_view(out + out_row + out_col * out_rows,
                       LhsTranspose ? ac : ar, n, out_rows);
  if constexpr (LhsTranspose) {
    if constexpr (Eye)
      O += alpha * A.transpose();
    else
      O.noalias() += alpha * A.transpose() *
                     Eigen::Map<const vector, RhsAlignment>(b + bk, n)
                         .asDiagonal();
  } else if constexpr (Eye) {
    O += alpha * A;
  } else {
    O.noalias() += alpha * A *
                   Eigen::Map<const vector, RhsAlignment>(b + bk, n)
                       .asDiagonal();
  }
}

template <bool LhsEye, bool RhsEye, int LhsAlignment = Eigen::Unaligned,
          int RhsAlignment = Eigen::Unaligned>
void pair_struct_struct(const double *a, size_t ak, const double *b, size_t bk,
                        double *out, size_t out_rows, size_t out_row,
                        size_t out_col, size_t n, double alpha) {
  Eigen::Map<vector, Eigen::Unaligned, Eigen::InnerStride<>> diagonal(
      out + out_row + out_col * out_rows, n,
      Eigen::InnerStride<>(out_rows + 1));
  if constexpr (LhsEye && RhsEye)
    diagonal.array() += alpha;
  else if constexpr (LhsEye)
    diagonal.array() +=
        alpha * Eigen::Map<const vector, RhsAlignment>(b + bk, n).array();
  else if constexpr (RhsEye)
    diagonal.array() +=
        alpha * Eigen::Map<const vector, LhsAlignment>(a + ak, n).array();
  else
    diagonal.array() +=
        alpha * Eigen::Map<const vector, LhsAlignment>(a + ak, n).array() *
        Eigen::Map<const vector, RhsAlignment>(b + bk, n).array();
}

#define MOTO_PAIR_WRAPPER(name, ...)                                           \
  extern "C" __attribute__((visibility("default"))) void name(                 \
      const double *a, size_t ar, size_t ac, size_t ald, size_t ak,             \
      const double *b, size_t br, size_t bc, size_t bld, size_t bk,            \
      double *out, size_t out_rows,                                             \
      size_t out_row, size_t out_col, size_t n, double alpha) {                \
    __VA_ARGS__;                                                               \
  }
#define MOTO_PAIR_DD_ONE(tag, lt, rt, suffix, la, ra)                          \
  MOTO_PAIR_WRAPPER(moto_linear_pair_dense_dense_##tag##_##suffix,             \
                    pair_dense_dense<lt, rt, la, ra>(                          \
                        a, ar, ac, ald, ak, b, br, bc, bld, bk, out,           \
                        out_rows, out_row,                                      \
                        out_col, n, alpha))
#define MOTO_PAIR_DD(tag, lt, rt)                                              \
  MOTO_PAIR_DD_ONE(tag, lt, rt, a_a, Eigen::Aligned, Eigen::Aligned)           \
  MOTO_PAIR_DD_ONE(tag, lt, rt, a_u, Eigen::Aligned, Eigen::Unaligned)         \
  MOTO_PAIR_DD_ONE(tag, lt, rt, u_a, Eigen::Unaligned, Eigen::Aligned)         \
  MOTO_PAIR_DD_ONE(tag, lt, rt, u_u, Eigen::Unaligned, Eigen::Unaligned)
MOTO_PAIR_DD(nn, false, false)
MOTO_PAIR_DD(tn, true, false)
MOTO_PAIR_DD(nt, false, true)
MOTO_PAIR_DD(tt, true, true)
#undef MOTO_PAIR_DD
#undef MOTO_PAIR_DD_ONE
#define MOTO_PAIR_SD(lp, le, tag, rt, align_tag, la, ra)                       \
  MOTO_PAIR_WRAPPER(moto_linear_pair_##lp##_dense_##tag##_##align_tag,         \
                    pair_struct_dense<le, rt, la, ra>(                         \
                        a, ak, b, br, bc, bld, bk, out, out_rows, out_row,     \
                        out_col,                                                \
                        n, alpha))
#define MOTO_PAIR_SD_ALL(lp, le, tag, rt)                                      \
  MOTO_PAIR_SD(lp, le, tag, rt, a_a, Eigen::Aligned, Eigen::Aligned)          \
  MOTO_PAIR_SD(lp, le, tag, rt, a_u, Eigen::Aligned, Eigen::Unaligned)        \
  MOTO_PAIR_SD(lp, le, tag, rt, u_a, Eigen::Unaligned, Eigen::Aligned)        \
  MOTO_PAIR_SD(lp, le, tag, rt, u_u, Eigen::Unaligned, Eigen::Unaligned)
MOTO_PAIR_SD_ALL(diag, false, n, false)
MOTO_PAIR_SD_ALL(diag, false, t, true)
MOTO_PAIR_SD(eye, true, n, false, x_a, Eigen::Unaligned, Eigen::Aligned)
MOTO_PAIR_SD(eye, true, n, false, x_u, Eigen::Unaligned, Eigen::Unaligned)
MOTO_PAIR_SD(eye, true, t, true, x_a, Eigen::Unaligned, Eigen::Aligned)
MOTO_PAIR_SD(eye, true, t, true, x_u, Eigen::Unaligned, Eigen::Unaligned)
#undef MOTO_PAIR_SD_ALL
#undef MOTO_PAIR_SD
#define MOTO_PAIR_DS(tag, lt, rp, re, align_tag, la, ra)                       \
  MOTO_PAIR_WRAPPER(moto_linear_pair_dense_##rp##_##tag##_##align_tag,         \
                    pair_dense_struct<lt, re, la, ra>(                         \
                        a, ar, ac, ald, ak, b, bk, out, out_rows, out_row,     \
                        out_col,                                                \
                        n, alpha))
#define MOTO_PAIR_DS_ALL(tag, lt, rp, re)                                      \
  MOTO_PAIR_DS(tag, lt, rp, re, a_a, Eigen::Aligned, Eigen::Aligned)          \
  MOTO_PAIR_DS(tag, lt, rp, re, a_u, Eigen::Aligned, Eigen::Unaligned)        \
  MOTO_PAIR_DS(tag, lt, rp, re, u_a, Eigen::Unaligned, Eigen::Aligned)        \
  MOTO_PAIR_DS(tag, lt, rp, re, u_u, Eigen::Unaligned, Eigen::Unaligned)
MOTO_PAIR_DS_ALL(n, false, diag, false)
MOTO_PAIR_DS_ALL(t, true, diag, false)
MOTO_PAIR_DS(n, false, eye, true, a_x, Eigen::Aligned, Eigen::Unaligned)
MOTO_PAIR_DS(n, false, eye, true, u_x, Eigen::Unaligned, Eigen::Unaligned)
MOTO_PAIR_DS(t, true, eye, true, a_x, Eigen::Aligned, Eigen::Unaligned)
MOTO_PAIR_DS(t, true, eye, true, u_x, Eigen::Unaligned, Eigen::Unaligned)
#undef MOTO_PAIR_DS_ALL
#undef MOTO_PAIR_DS
#define MOTO_PAIR_SS(lp, le, rp, re, align_tag, la, ra)                        \
  MOTO_PAIR_WRAPPER(moto_linear_pair_##lp##_##rp##_##align_tag,                \
                    pair_struct_struct<le, re, la, ra>(                        \
                        a, ak, b, bk, out, out_rows, out_row, out_col, n,      \
                        alpha))
MOTO_PAIR_SS(diag, false, diag, false, a_a, Eigen::Aligned, Eigen::Aligned)
MOTO_PAIR_SS(diag, false, diag, false, a_u, Eigen::Aligned, Eigen::Unaligned)
MOTO_PAIR_SS(diag, false, diag, false, u_a, Eigen::Unaligned, Eigen::Aligned)
MOTO_PAIR_SS(diag, false, diag, false, u_u, Eigen::Unaligned, Eigen::Unaligned)
MOTO_PAIR_SS(diag, false, eye, true, a_x, Eigen::Aligned, Eigen::Unaligned)
MOTO_PAIR_SS(diag, false, eye, true, u_x, Eigen::Unaligned, Eigen::Unaligned)
MOTO_PAIR_SS(eye, true, diag, false, x_a, Eigen::Unaligned, Eigen::Aligned)
MOTO_PAIR_SS(eye, true, diag, false, x_u, Eigen::Unaligned, Eigen::Unaligned)
MOTO_PAIR_SS(eye, true, eye, true, x_x, Eigen::Unaligned, Eigen::Unaligned)
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

product_kernel compile_product(product_spec spec,
                               const std::filesystem::path &cache_dir) {
  auto function = reinterpret_cast<product_kernel::function_type>(
      compile_source(emit_product_function(spec, symbol_name, true), cache_dir));
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
                << panel.cols << ',' << panel.transposed << ','
                << panel.storage_offset << ',' << panel.storage_rows << ';';
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

graph_kernel::graph_kernel(
    std::unique_ptr<detail::casadi_mx_graph_instance> instance)
    : instance_(std::move(instance)) {
  if (instance_) {
    inputs_ = detail::casadi_mx_graph_inputs(*instance_->plan);
    outputs_ = detail::casadi_mx_graph_outputs(*instance_->plan);
    entries_ = detail::casadi_mx_graph_entries(*instance_->plan);
  }
}
graph_kernel::graph_kernel() = default;
graph_kernel::graph_kernel(graph_kernel &&) noexcept = default;
graph_kernel &graph_kernel::operator=(graph_kernel &&) noexcept = default;
graph_kernel::~graph_kernel() = default;
size_t graph_kernel::input_count() const { return inputs_; }

graph_kernel graph_kernel::instantiate(
    std::vector<sparse_matrix> *workspace) const {
  if (!instance_) return {};
  return graph_kernel(std::make_unique<detail::casadi_mx_graph_instance>(
      instance_->plan, workspace));
}

void graph_kernel::operator()(std::span<scalar_t *> pointers) const {
  (*this)(0, pointers);
}

void graph_kernel::operator()(size_t entry,
                              std::span<scalar_t *> pointers) const {
  if (!instance_ || pointers.size() != pointer_count())
    throw std::invalid_argument("invalid linear graph kernel invocation");
  instance_->run(entry, pointers);
}
size_t graph_kernel::entry_count() const { return entries_; }

graph_kernel compile_graph(const std::vector<casadi::MX> &inputs,
                           const std::vector<casadi::MX> &outputs,
                           std::vector<sparse_matrix> *workspace,
                           const std::filesystem::path &cache_dir) {
  return compile_graph(inputs, std::vector<std::vector<casadi::MX>>{outputs},
                       workspace, cache_dir);
}

graph_kernel compile_graph(
    const std::vector<casadi::MX> &inputs,
    const std::vector<std::vector<casadi::MX>> &output_entries,
    std::vector<sparse_matrix> *workspace,
    const std::filesystem::path &cache_dir) {
  return compile_graph(inputs, output_entries, {}, workspace, cache_dir);
}

graph_kernel compile_graph(
    const std::vector<casadi::MX> &inputs,
    const std::vector<std::vector<casadi::MX>> &output_entries,
    std::span<const matrix_layout> input_layouts,
    std::vector<sparse_matrix> *workspace,
    const std::filesystem::path &cache_dir) {
  return compile_graph({}, inputs, output_entries, input_layouts, workspace,
                       cache_dir, {});
}

graph_kernel compile_graph(
    std::string_view artifact_identity,
    const std::vector<casadi::MX> &inputs,
    const std::vector<std::vector<casadi::MX>> &output_entries,
    std::span<const matrix_layout> input_layouts,
    std::vector<sparse_matrix> *workspace,
    const std::filesystem::path &cache_dir,
    std::span<const casadi::MX> spd_factors) {
  std::vector<casadi::MX> raw_outputs;
  std::vector<size_t> entry_outputs;
  for (const auto &entry : output_entries) {
    entry_outputs.push_back(entry.size());
    raw_outputs.insert(raw_outputs.end(), entry.begin(), entry.end());
  }
  if (raw_outputs.empty()) return {};
  raw_outputs.insert(raw_outputs.end(), spd_factors.begin(),
                     spd_factors.end());
  const bool named_artifact = !artifact_identity.empty();
  std::string serialized = named_artifact ? "casadi_mx_translator_named_v4:"
                                          : "casadi_mx_translator_v6:";
  if (artifact_identity.empty()) {
    const casadi::Function raw_function(
        "moto_casadi_linear_graph_input", inputs, raw_outputs);
    serialized += raw_function.serialize();
  } else {
    serialized += artifact_identity;
    for (const auto &input : inputs)
      serialized += ":in:" + std::to_string(input.size1()) + ',' +
                    std::to_string(input.size2()) + ',' +
                    std::to_string(input.nnz());
    for (const auto &output : raw_outputs)
      serialized += ":out:" + std::to_string(output.size1()) + ',' +
                    std::to_string(output.size2()) + ',' +
                    std::to_string(output.nnz());
  }
  for (const size_t count : entry_outputs)
    serialized += ':' + std::to_string(count);
  serialized += ":spd:" + std::to_string(spd_factors.size());
  if (!named_artifact && !spd_factors.empty())
    serialized += casadi::Function(
        "moto_spd_properties", inputs,
        std::vector<casadi::MX>(spd_factors.begin(), spd_factors.end()))
                      .serialize();
  for (const auto &layout : input_layouts) {
    serialized += ":layout:" + std::to_string(layout.rows) + ':' +
                  std::to_string(layout.cols);
    for (const auto &panel : layout.panels)
      serialized += ':' + std::to_string(static_cast<int>(panel.pattern)) +
                    ',' + std::to_string(panel.row_offset) + ',' +
                    std::to_string(panel.col_offset) + ',' +
                    std::to_string(panel.rows) + ',' +
                    std::to_string(panel.cols) + ',' +
                    std::to_string(panel.transposed) + ',' +
                    std::to_string(panel.storage_offset) + ',' +
                    std::to_string(panel.storage_rows);
  }
  const std::string key = utils::compute_md5_from_bytes(serialized);
  std::shared_ptr<const detail::casadi_mx_graph_plan> plan;
  std::shared_ptr<std::mutex> key_mutex;
  {
    std::lock_guard lock(graph_mutex);
    if (auto found = graph_plans.find(key); found != graph_plans.end())
      plan = found->second.lock();
    if (plan)
      return graph_kernel(std::make_unique<detail::casadi_mx_graph_instance>(
          std::move(plan), workspace));
    auto &slot = graph_compile_locks[key];
    if (!slot)
      slot = std::make_shared<std::mutex>();
    key_mutex = slot;
  }
  std::lock_guard key_lock(*key_mutex);
  {
    std::lock_guard lock(graph_mutex);
    if (auto found = graph_plans.find(key); found != graph_plans.end())
      plan = found->second.lock();
  }
  if (!plan) {
    std::filesystem::create_directories(cache_dir);
    const auto plan_cache = cache_dir / ("plan_v13_" + key + ".cbor");
    const auto mx_cache = cache_dir / ("graph_" + key + ".casadi");
    if (std::filesystem::exists(plan_cache)) {
      try {
        plan = detail::load_casadi_mx_graph_plan(plan_cache, cache_dir);
      } catch (const std::exception &) {
        std::error_code ec;
        std::filesystem::remove(plan_cache, ec);
      }
    }
    if (!plan) {
      casadi::Function function;
      if (std::filesystem::exists(mx_cache)) {
        try {
          std::ifstream input(mx_cache, std::ios::binary);
          function = casadi::Function::deserialize(input);
        } catch (const std::exception &) {
          std::error_code ec;
          std::filesystem::remove(mx_cache, ec);
        }
      }
      if (function.is_null()) {
        auto graph_entries = output_entries;
        if (!spd_factors.empty())
          graph_entries.emplace_back(spd_factors.begin(), spd_factors.end());
        const auto batched_entries = batch_independent_mx_products(
            batch_independent_mx_solves(graph_entries));
        std::vector<casadi::MX> outputs;
        for (const auto &entry : batched_entries)
          outputs.insert(outputs.end(), entry.begin(), entry.end());
        outputs = casadi::MX::cse(outputs);
        function = casadi::Function("moto_casadi_linear_graph", inputs,
                                    outputs);
        const auto tmp =
            mx_cache.string() + ".tmp." + std::to_string(::getpid());
        try {
          {
            std::ofstream output(tmp, std::ios::binary);
            if (!output)
              throw std::runtime_error("failed to create MX graph cache");
            function.serialize(output);
          }
          std::filesystem::rename(tmp, mx_cache);
        } catch (...) {
          std::error_code ec;
          std::filesystem::remove(tmp, ec);
          throw;
        }
      }
      plan = detail::translate_casadi_mx_graph(
          function, entry_outputs, input_layouts, cache_dir,
          spd_factors.size(), plan_cache);
    }
    std::lock_guard lock(graph_mutex);
    graph_plans[key] = plan;
  }
  return graph_kernel(std::make_unique<detail::casadi_mx_graph_instance>(
      std::move(plan), workspace));
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

struct cached_dense_write {
  size_t out_rows = 0;
  scalar_t alpha = 0.;
  bool overwrite = false;
  batch_product_kernel kernel;
  std::vector<scalar_t *> pointers;
};

struct matrix_cache {
  std::mutex mutex;
  std::vector<cached_product> products;
  std::vector<cached_sparse_product> sparse_products;
  std::vector<cached_dense_write> dense_writes;
};

matrix_layout describe(const casadi::Sparsity &sp) {
  const size_t rows = sp.size1(), cols = sp.size2();
  matrix_layout result{rows, cols, {}};
  if (!sp.nnz()) return result;
  if (static_cast<size_t>(sp.nnz()) == rows * cols) {
    result.panels.push_back({sparsity::dense, 0, 0, rows, cols});
    return result;
  }
  std::vector<std::vector<size_t>> by_row(rows);
  const auto &colind = sp.get_colind();
  const auto &row = sp.get_row();
  for (size_t col = 0; col < cols; ++col)
    for (casadi_int nz = colind[col]; nz < colind[col + 1]; ++nz)
      by_row[row[nz]].push_back(col);

  struct active_rectangle { size_t block, last_row; };
  std::vector<sparse_block_spec> blocks;
  std::map<std::pair<size_t, size_t>, active_rectangle> active;
  for (size_t r = 0; r < rows; ++r) {
    std::map<std::pair<size_t, size_t>, active_rectangle> next;
    for (size_t i = 0; i < by_row[r].size();) {
      const size_t begin = by_row[r][i];
      size_t end = begin + 1;
      while (++i < by_row[r].size() && by_row[r][i] == end) ++end;
      const auto key = std::pair{begin, end};
      if (const auto found = active.find(key);
          found != active.end() && found->second.last_row + 1 == r) {
        ++blocks[found->second.block].rows;
        next.emplace(key, active_rectangle{found->second.block, r});
      } else {
        blocks.push_back({r, begin, 1, end - begin, sparsity::dense});
        next.emplace(key, active_rectangle{blocks.size() - 1, r});
      }
    }
    active = std::move(next);
  }
  std::map<ptrdiff_t, std::vector<size_t>> diagonals;
  for (size_t i = 0; i < blocks.size(); ++i)
    if (blocks[i].rows == 1 && blocks[i].cols == 1)
      diagonals[static_cast<ptrdiff_t>(blocks[i].col) - blocks[i].row]
          .push_back(i);
  std::vector<bool> replaced(blocks.size());
  for (const auto &[offset, candidates] : diagonals) {
    (void)offset;
    for (size_t begin = 0; begin < candidates.size();) {
      size_t end = begin + 1;
      while (end < candidates.size() &&
             blocks[candidates[end]].row ==
                 blocks[candidates[end - 1]].row + 1 &&
             blocks[candidates[end]].col ==
                 blocks[candidates[end - 1]].col + 1)
        ++end;
      if (end - begin > 1) {
        auto diagonal = blocks[candidates[begin]];
        diagonal.rows = diagonal.cols = end - begin;
        diagonal.pattern = sparsity::diag;
        blocks.push_back(diagonal);
        for (size_t i = begin; i < end; ++i) replaced[candidates[i]] = true;
      }
      begin = end;
    }
  }
  std::vector<sparse_block_spec> canonical;
  for (size_t i = 0; i < blocks.size(); ++i)
    if (i >= replaced.size() || !replaced[i]) canonical.push_back(blocks[i]);
  const auto plan = make_sparse_layout_plan(canonical);
  for (const auto &panel : plan.panels)
    result.panels.push_back({panel.pattern, panel.row, panel.col,
                             panel.rows, panel.cols});
  std::ranges::stable_sort(result.panels, {}, [](const panel_layout &panel) {
    return panel.pattern == sparsity::dense ? 0
           : panel.pattern == sparsity::diag ? 1 : 2;
  });
  return result;
}

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
  if (sparse.diagonal_segments_.empty()) {
    add(sparse.diag_panels_, sparsity::diag);
  } else {
    for (const auto &p : sparse.diagonal_segments_)
      layout.panels.push_back(
          {sparsity::diag, p.row, p.col, p.rows, p.cols});
  }
  add(sparse.eye_panels_, sparse.dynamic_eye_ ? sparsity::diag : sparsity::eye);
  return layout;
}

std::vector<scalar_t *> panel_pointers(const ::moto::sparse_matrix &sparse) {
  std::vector<scalar_t *> pointers;
  const auto add = [&](const auto &panels) {
    for (const auto &p : panels)
      pointers.push_back(const_cast<scalar_t *>(p.data_.data()));
  };
  add(sparse.dense_panels_);
  if (sparse.diagonal_segments_.empty()) {
    add(sparse.diag_panels_);
  } else {
    for (const auto &p : sparse.diagonal_segments_)
      pointers.push_back(const_cast<scalar_t *>(
          sparse.diag_panels_[p.storage_panel].data_.data() +
          p.storage_offset));
  }
  add(sparse.eye_panels_);
  return pointers;
}

void rowwise_kernel::operator()(std::span<scalar_t *const> panels,
                                const scalar_t *scale,
                                scalar_t *output) const {
  if (!function_ || panels.size() != panels_)
    throw std::invalid_argument("invalid sparse JIT rowwise invocation");
  function_(panels.data(), scale, output);
}

namespace {
std::string emit_rowwise_function(const matrix_layout &layout, rowwise_op op,
                                  std::string_view name) {
  if (!layout.rows || layout.panels.empty())
    throw std::invalid_argument("empty sparse JIT rowwise profile");
  std::ostringstream source;
  source << "#include <cstddef>\n";
  if (op == rowwise_op::scale) {
    if (std::ranges::any_of(layout.panels, [](const panel_layout &panel) {
          return panel.pattern == sparsity::eye;
        }))
      throw std::invalid_argument("row scaling requires dynamic eye profile");
    source << "extern \"C\" void moto_linear_row_scale_dense(double*,"
              "std::size_t,std::size_t,const double*,std::size_t);\n"
              "extern \"C\" void moto_linear_row_scale_diag(double*,"
              "std::size_t,const double*,std::size_t);\n";
  } else {
    const auto prefix =
        op == rowwise_op::scaled_inf_norm ? "scaled_row_" : "row_";
    for (const auto pattern : {"dense", "diag", "eye"})
      source << "extern \"C\" void moto_linear_" << prefix << "infnorm_"
             << pattern
             << "(const double*,std::size_t,std::size_t,const double*,double*,"
                "std::size_t);\n";
  }
  source << "static void " << name
         << "(double *const *p,const double*s,double*out){\n";
  for (size_t i = 0; i < layout.panels.size(); ++i) {
    const auto &panel = layout.panels[i];
    if (panel.row_offset + panel.rows > layout.rows)
      throw std::invalid_argument("invalid sparse JIT rowwise panel");
    const auto pattern = panel.pattern == sparsity::dense ? "dense" :
                         panel.pattern == sparsity::diag  ? "diag" : "eye";
    if (op == rowwise_op::scale) {
      source << "  moto_linear_row_scale_" << pattern << "(p[" << i << "],"
             << panel.rows << ',';
      if (panel.pattern == sparsity::dense)
        source << panel.cols << ',';
      source << "s," << panel.row_offset << ");\n";
    } else {
      source << "  moto_linear_"
             << (op == rowwise_op::scaled_inf_norm ? "scaled_row_" : "row_")
             << "infnorm_" << pattern << "(p[" << i << "]," << panel.rows
             << ',' << panel.cols << ",s,out," << panel.row_offset << ");\n";
    }
  }
  source << "}\n";
  return source.str();
}
} // namespace

rowwise_kernels compile_rowwise(matrix_layout layout,
                                const std::filesystem::path &cache_dir) {
  std::ostringstream source;
  const std::array ops{rowwise_op::scale, rowwise_op::inf_norm,
                       rowwise_op::scaled_inf_norm};
  for (size_t i = 0; i < ops.size(); ++i)
    source << emit_rowwise_function(layout, ops[i],
                                    "moto_linear_jit_rowwise_" +
                                        std::to_string(i));
  source << "extern \"C\" __attribute__((visibility(\"default\"))) void *"
         << symbol_name << "(std::size_t i){static void*f[]={";
  for (size_t i = 0; i < ops.size(); ++i) {
    if (i)
      source << ',';
    source << "reinterpret_cast<void*>(&moto_linear_jit_rowwise_" << i << ')';
  }
  source << "};return f[i];}\n";
  using registry_type = void *(*)(size_t);
  auto registry = reinterpret_cast<registry_type>(
      compile_source(source.str(), cache_dir));
  const auto make = [&](size_t index) {
    return rowwise_kernel(
        layout.panels.size(),
        reinterpret_cast<rowwise_kernel::function_type>(registry(index)));
  };
  return {make(0), make(1), make(2)};
}

void run_product(const ::moto::sparse_matrix &sparse, product_op op,
                 scalar_t sign, const scalar_t *other, size_t other_rows,
                 size_t other_cols, scalar_t *out, size_t out_rows,
                 size_t out_cols) {
  if (sparse.is_empty() || !other_rows || !other_cols || !out_rows ||
      !out_cols)
    return;
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
  bool operation_transpose;
  bool transpose() const { return panel.transposed != operation_transpose; }
  size_t row() const {
    return operation_transpose ? panel.col_offset : panel.row_offset;
  }
  size_t col() const {
    return operation_transpose ? panel.row_offset : panel.col_offset;
  }
  size_t rows() const { return operation_transpose ? panel.cols : panel.rows; }
  size_t cols() const { return operation_transpose ? panel.rows : panel.cols; }
  size_t physical_rows() const {
    return panel.transposed ? panel.cols : panel.rows;
  }
  size_t physical_cols() const {
    return panel.transposed ? panel.rows : panel.cols;
  }
  size_t leading_rows() const {
    return panel.storage_rows
               ? panel.storage_rows
               : physical_rows();
  }
  bool structured() const { return panel.pattern != sparsity::dense; }
};

std::string emit_sparse_product_source(const matrix_layout &lhs,
                                       const matrix_layout &rhs,
                                       bool lhs_transpose, scalar_t sign,
                                       size_t out_rows) {
  std::ostringstream s;
  const auto pattern_name = [](sparsity p) {
    return p == sparsity::dense  ? "dense"
           : p == sparsity::diag ? "diag"
                                 : "eye";
  };
  const auto alignment_tag = [](size_t offset) {
    return offset * sizeof(scalar_t) % EIGEN_MAX_ALIGN_BYTES == 0 ? 'a' : 'u';
  };
  const auto helper_name = [&](const effective_panel &l,
                               const effective_panel &r, size_t lk,
                               size_t rk) {
    std::string name = "moto_linear_pair_";
    name += pattern_name(l.panel.pattern);
    name += '_';
    name += pattern_name(r.panel.pattern);
    if (!l.structured() || !r.structured()) {
      name += '_';
      if (!l.structured())
        name += l.transpose() ? 't' : 'n';
      if (!r.structured())
        name += r.transpose() ? 't' : 'n';
    }
    const auto view_alignment = [&](const effective_panel &p, size_t k,
                                    bool lhs) {
      if (p.panel.pattern == sparsity::eye)
        return 'x';
      size_t offset = k;
      if (p.panel.pattern == sparsity::dense) {
        if (lhs)
          offset = p.transpose() ? k : k * p.leading_rows();
        else
          offset = p.transpose() ? k * p.leading_rows() : k;
      }
      return alignment_tag(p.panel.storage_offset + offset);
    };
    name += '_';
    name += view_alignment(l, lk, true);
    name += '_';
    name += view_alignment(r, rk, false);
    return name;
  };
  s << "#include <cstddef>\n";
  for (const auto &lp : lhs.panels) for (const auto &rp : rhs.panels) {
      const effective_panel l{lp, lhs_transpose}, r{rp, false};
      const size_t begin = std::max(l.col(), r.row());
      const size_t end = std::min(l.col() + l.cols(), r.row() + r.rows());
      if (end <= begin)
        continue;
      const size_t lk = begin - l.col(), rk = begin - r.row();
      s << "extern \"C\" void " << helper_name(l, r, lk, rk)
        << "(const double*,std::size_t,std::size_t,std::size_t,std::size_t,"
           "const double*,std::size_t,std::size_t,std::size_t,std::size_t,"
           "double*,std::size_t,"
           "std::size_t,std::size_t,std::size_t,double);\n";
    }
  s << "extern \"C\" __attribute__((visibility(\"default\"))) void "
    << symbol_name << "(double *const *p) {\n";
  const size_t output_slot = lhs.panels.size() + rhs.panels.size();
  for (size_t li = 0; li < lhs.panels.size(); ++li) {
    const effective_panel l{lhs.panels[li], lhs_transpose};
    for (size_t ri = 0; ri < rhs.panels.size(); ++ri) {
      const effective_panel r{rhs.panels[ri], false};
      const size_t begin = std::max(l.col(), r.row());
      const size_t end = std::min(l.col() + l.cols(), r.row() + r.rows());
      if (end <= begin)
        continue;
      const size_t n = end - begin;
      const size_t lk = begin - l.col(), rk = begin - r.row();
      const size_t rslot = lhs.panels.size() + ri;
      const size_t out_row = l.row() + (l.structured() ? lk : 0);
      const size_t out_col = r.col() + (r.structured() ? rk : 0);
      s << "  " << helper_name(l, r, lk, rk) << "(p[" << li
        << "]+"
        << l.panel.storage_offset << ',' << l.physical_rows()
        << ',' << l.physical_cols() << ',' << l.leading_rows() << ',' << lk
        << ",p[" << rslot << "]+"
        << r.panel.storage_offset << ',' << r.physical_rows() << ','
        << r.physical_cols() << ',' << r.leading_rows() << ',' << rk << ",p["
        << output_slot << "]," << out_rows << ',' << out_row << ',' << out_col << ',' << n
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
  const auto source = emit_sparse_product_source(a, b, lhs_t, sign, out_rows);
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
  const auto matches = [&](const cached_dense_write &entry) {
    return entry.out_rows == out_rows && entry.alpha == alpha &&
           entry.overwrite == overwrite;
  };
  if (auto found = std::ranges::find_if(cache.dense_writes, matches);
      found != cache.dense_writes.end()) {
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
  cache.dense_writes.push_back(
      {out_rows, alpha, overwrite, std::move(kernel), std::move(pointers)});
}

void prepare_dense_write(const sparse_matrix &sparse, size_t out_rows,
                         scalar_t alpha, bool overwrite) {
  if (sparse.is_empty() || !out_rows)
    return;
  run_dense_write(sparse, nullptr, out_rows, alpha, overwrite);
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

size_t condensation_spec::pointer_count() const {
  return 2 * argument_count() + 2 * residual_signs.size() +
         condensation_pairs(*this).size();
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
    offset += spec.constraints[i].pointer_count();
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
