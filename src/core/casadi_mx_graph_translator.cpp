#include "casadi_mx_graph_translator.hpp"

#include <moto/utils/codegen.hpp>

#include <Eigen/Cholesky>
#include <Eigen/LU>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace moto::linear_backend::detail {
struct cached_factor {
  Eigen::PartialPivLU<matrix> lu;
  Eigen::LLT<matrix> llt;
  size_t epoch = 0;
  bool spd = false;

  template <typename value_type>
  void compute(value_type &value, bool use_spd) {
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
  template <typename rhs_type, typename result_type>
  void solve(const rhs_type &rhs, result_type &result) {
    if (spd) result = llt.solve(rhs);
    else result = lu.solve(rhs);
  }
  template <typename rhs_type, typename result_type>
  void transpose_solve(const rhs_type &rhs, result_type &result) {
    if (spd) result = llt.solve(rhs);
    else result = lu.transpose().solve(rhs);
  }
  template <typename result_type>
  void inverse(result_type &result) {
    if (spd) result = llt.solve(matrix::Identity(llt.rows(), llt.cols()));
    else result = lu.inverse();
  }
};

namespace {

using index_t = casadi_int;
constexpr size_t no_value = std::numeric_limits<size_t>::max();

struct sparse_index {
  explicit sparse_index(const casadi::Sparsity &value)
      : rows(value.size1()), cols(value.size2()), colind(value.get_colind()),
        row(value.get_row()) {
    for (index_t col = 0; col < cols; ++col)
      for (index_t nz = colind[col]; nz < colind[col + 1]; ++nz)
        lookup.emplace(row[nz] + rows * col, nz);
  }
  index_t find(index_t r, index_t c) const {
    const auto it = lookup.find(r + rows * c);
    return it == lookup.end() ? -1 : it->second;
  }
  index_t rows = 0, cols = 0;
  std::vector<index_t> colind, row;
  std::unordered_map<index_t, index_t> lookup;
};

struct nz_address {
  size_t panel = no_value;
  size_t offset = 0;
};

struct value_layout {
  casadi::Sparsity sparsity;
  matrix_layout matrix;
  std::vector<nz_address> csc;
  std::string fingerprint;
  bool allocate = true;
  /// Semantic sparsity may be denser than physical storage.  This is used for
  /// MX project/densify nodes: missing physical entries are immutable zeros,
  /// not a reason to allocate and clear a dense temporary.
  bool virtual_zeros = false;
  uint64_t local_entries = 0;

  bool direct_csc_panel() const {
    if (matrix.panels.size() != 1 ||
        matrix.panels[0].transposed || matrix.panels[0].storage_offset ||
        matrix.panels[0].pattern == sparsity::eye ||
        csc.size() != (matrix.panels[0].pattern == sparsity::dense
                          ? matrix.panels[0].rows * matrix.panels[0].cols
                          : matrix.panels[0].rows))
      return false;
    for (size_t i = 0; i < csc.size(); ++i)
      if (csc[i].panel != 0 || csc[i].offset != i) return false;
    return true;
  }
};

value_layout make_layout(const casadi::Sparsity &sp) {
  value_layout result{.sparsity = sp,
                      .matrix = {.rows = static_cast<size_t>(sp.size1()),
                                 .cols = static_cast<size_t>(sp.size2())}};
  sparse_pattern pattern{static_cast<size_t>(sp.size1()),
                         static_cast<size_t>(sp.size2())};
  for (const auto value : sp.get_colind())
    pattern.colind.push_back(static_cast<size_t>(value));
  for (const auto value : sp.get_row())
    pattern.row.push_back(static_cast<size_t>(value));
  result.matrix = panelize_pattern(pattern);
  const sparse_index index(sp);
  result.csc.resize(sp.nnz());
  std::vector<bool> assigned(sp.nnz());
  for (size_t panel_index = 0; panel_index < result.matrix.panels.size();
       ++panel_index) {
    const auto &panel = result.matrix.panels[panel_index];
    if (panel.pattern == sparsity::dense) {
      for (size_t col = 0; col < panel.cols; ++col)
        for (size_t row = 0; row < panel.rows; ++row) {
          const index_t nz = index.find(panel.row_offset + row,
                                        panel.col_offset + col);
          if (nz < 0) continue;
          const size_t leading = panel.storage_rows
                                     ? panel.storage_rows
                                     : (panel.transposed ? panel.cols
                                                         : panel.rows);
          const size_t offset = panel.storage_offset +
              (panel.transposed ? col + row * leading
                                : row + col * leading);
          result.csc[nz] = {panel_index, offset};
          assigned[nz] = true;
        }
    } else {
      for (size_t k = 0; k < panel.rows; ++k) {
        const index_t nz = index.find(panel.row_offset + k,
                                      panel.col_offset + k);
        if (nz < 0)
          throw std::runtime_error(
              "graph diagonal panel covers a structural zero");
        result.csc[nz] = {panel_index, panel.storage_offset + k};
        assigned[nz] = true;
      }
    }
  }
  if (std::ranges::find(assigned, false) != assigned.end())
    throw std::runtime_error("graph panel layout does not cover its sparsity");
  return result;
}

void rebuild_csc_map(value_layout &layout) {
  const sparse_index index(layout.sparsity);
  layout.csc.assign(layout.sparsity.nnz(), {});
  std::vector<bool> assigned(layout.sparsity.nnz());
  for (size_t pi = 0; pi < layout.matrix.panels.size(); ++pi) {
    const auto &panel = layout.matrix.panels[pi];
    if (panel.pattern == sparsity::dense) {
      for (size_t col = 0; col < panel.cols; ++col)
        for (size_t row = 0; row < panel.rows; ++row) {
          const index_t nz = index.find(panel.row_offset + row,
                                        panel.col_offset + col);
          if (nz < 0 || assigned[nz]) continue;
          const size_t leading = panel.storage_rows
                                     ? panel.storage_rows
                                     : (panel.transposed ? panel.cols
                                                         : panel.rows);
          const size_t offset = panel.storage_offset +
              (panel.transposed ? col + row * leading
                                : row + col * leading);
          layout.csc[nz] = {pi, offset};
          assigned[nz] = true;
        }
    } else {
      for (size_t k = 0; k < panel.rows; ++k) {
        const index_t nz = index.find(panel.row_offset + k,
                                      panel.col_offset + k);
        if (nz < 0 || assigned[nz]) continue;
        layout.csc[nz] = {pi, panel.storage_offset + k};
        assigned[nz] = true;
      }
    }
  }
  if (!layout.virtual_zeros &&
      std::ranges::find(assigned, false) != assigned.end())
    throw std::runtime_error("graph panel layout does not cover its sparsity");
}

bool contains_sparsity(const casadi::Sparsity &outer,
                       const casadi::Sparsity &inner) {
  if (outer.size1() != inner.size1() || outer.size2() != inner.size2())
    return false;
  const sparse_index out(outer), in(inner);
  for (index_t col = 0; col < in.cols; ++col)
    for (index_t nz = in.colind[col]; nz < in.colind[col + 1]; ++nz)
      if (out.find(in.row[nz], col) < 0) return false;
  return true;
}

void ensure_product_coverage(value_layout &output, const value_layout &lhs,
                             const value_layout &rhs) {
  struct block { size_t row, col, rows, cols; bool diagonal; };
  std::vector<block> contributions;
  for (const auto &l : lhs.matrix.panels)
    for (const auto &r : rhs.matrix.panels) {
      const size_t begin = std::max(l.col_offset, r.row_offset);
      const size_t end = std::min(l.col_offset + l.cols,
                                  r.row_offset + r.rows);
      if (end <= begin) continue;
      const size_t n = end - begin;
      const bool ls = l.pattern != sparsity::dense;
      const bool rs = r.pattern != sparsity::dense;
      contributions.push_back({
          l.row_offset + (ls ? begin - l.col_offset : 0),
          r.col_offset + (rs ? begin - r.row_offset : 0),
          ls ? n : l.rows, rs ? n : r.cols, ls && rs});
    }
  const auto contains = [](const panel_layout &panel, const block &value) {
    if (panel.pattern == sparsity::dense)
      return panel.row_offset <= value.row && panel.col_offset <= value.col &&
             value.row + value.rows <= panel.row_offset + panel.rows &&
             value.col + value.cols <= panel.col_offset + panel.cols;
    return value.diagonal && panel.pattern == sparsity::diag &&
           panel.row_offset <= value.row && panel.col_offset <= value.col &&
           value.row - panel.row_offset == value.col - panel.col_offset &&
           value.row + value.rows <= panel.row_offset + panel.rows &&
           value.col + value.cols <= panel.col_offset + panel.cols;
  };
  const auto overlaps = [](const panel_layout &a, const panel_layout &b) {
    return a.row_offset < b.row_offset + b.rows &&
           b.row_offset < a.row_offset + a.rows &&
           a.col_offset < b.col_offset + b.cols &&
           b.col_offset < a.col_offset + a.cols;
  };
  for (const auto &value : contributions) {
    if (std::ranges::any_of(output.matrix.panels,
                            [&](const auto &p) { return contains(p, value); }))
      continue;
    panel_layout merged{sparsity::dense, value.row, value.col, value.rows,
                        value.cols};
    bool changed;
    do {
      changed = false;
      for (size_t i = 0; i < output.matrix.panels.size();) {
        const auto &panel = output.matrix.panels[i];
        if (!overlaps(merged, panel)) {
          ++i;
          continue;
        }
        const size_t row_end = std::max(merged.row_offset + merged.rows,
                                        panel.row_offset + panel.rows);
        const size_t col_end = std::max(merged.col_offset + merged.cols,
                                        panel.col_offset + panel.cols);
        merged.row_offset = std::min(merged.row_offset, panel.row_offset);
        merged.col_offset = std::min(merged.col_offset, panel.col_offset);
        merged.rows = row_end - merged.row_offset;
        merged.cols = col_end - merged.col_offset;
        output.matrix.panels.erase(output.matrix.panels.begin() + i);
        changed = true;
      }
    } while (changed);
    output.matrix.panels.push_back(merged);
  }
  std::ranges::stable_sort(output.matrix.panels, {},
                           [](const panel_layout &panel) {
    return panel.pattern == sparsity::dense ? 0
           : panel.pattern == sparsity::diag ? 1
                                             : 2;
  });
  rebuild_csc_map(output);
}

sparse_matrix make_storage(const value_layout &layout) {
  sparse_matrix result;
  result.resize(layout.matrix.rows, layout.matrix.cols);
  for (const auto &panel : layout.matrix.panels) {
    if (panel.storage_offset)
      throw std::logic_error(
          "offset graph view panel cannot own independent sparse storage");
    result.insert(panel.row_offset, panel.col_offset, panel.rows, panel.cols,
                  panel.pattern);
  }
  return result;
}

std::vector<index_t> slice(const casadi::Dict &value) {
  const auto get = [&](std::string_view key) {
    const auto found = value.find(std::string(key));
    if (found == value.end())
      throw std::runtime_error("incomplete CasADi slice metadata");
    return found->second.to_int();
  };
  const index_t start = get("start"), stop = get("stop"), step = get("step");
  if (!step) throw std::runtime_error("invalid zero CasADi slice step");
  std::vector<index_t> result;
  if (step > 0)
    for (index_t i = start; i < stop; i += step) result.push_back(i);
  else
    for (index_t i = start; i > stop; i += step) result.push_back(i);
  return result;
}

std::vector<index_t> nonzero_indices(const casadi::Dict &info,
                                     size_t expected) {
  if (const auto found = info.find("nz"); found != info.end())
    return found->second.to_int_vector();
  if (const auto found = info.find("slice"); found != info.end())
    return slice(found->second.to_dict());
  const auto inner = info.find("inner"), outer = info.find("outer");
  if (inner != info.end() && outer != info.end()) {
    const auto i = slice(inner->second.to_dict());
    const auto o = slice(outer->second.to_dict());
    std::vector<index_t> result;
    result.reserve(i.size() * o.size());
    for (const index_t ov : o)
      for (const index_t iv : i) result.push_back(ov + iv);
    if (result.size() == expected) return result;
    result.clear();
    for (const index_t iv : i)
      for (const index_t ov : o) result.push_back(ov + iv);
    return result;
  }
  throw std::runtime_error("unsupported CasADi nonzero-index metadata");
}

std::vector<index_t> coordinate_map(const casadi::Sparsity &output,
                                    const casadi::Sparsity &input) {
  const sparse_index out(output), in(input);
  if (in.rows == 1 && in.cols == 1 && in.row.size() == 1)
    return std::vector<index_t>(out.row.size(), 0);
  std::vector<index_t> map;
  map.reserve(out.row.size());
  for (index_t col = 0; col < out.cols; ++col)
    for (index_t nz = out.colind[col]; nz < out.colind[col + 1]; ++nz)
      map.push_back(in.find(out.row[nz], col));
  return map;
}

struct execution_context;

struct operation {
  uint64_t entries = 0;
  virtual ~operation() = default;
  virtual void execute(execution_context &) const = 0;
  virtual bool binding_only() const { return false; }
};

struct casadi_mx_graph_plan_impl;

struct execution_context {
  const casadi_mx_graph_plan_impl &plan;
  const casadi_mx_graph_instance &instance;
  std::span<scalar_t *> external;
  size_t entry;

  scalar_t read(size_t value, index_t nz) const;
  void write(size_t value, size_t nz, scalar_t scalar) const;
  void zero(size_t value) const;
  std::vector<scalar_t *> &pointers(size_t value) const {
    return instance.slot_pointers[value];
  }
};

struct casadi_mx_graph_plan_impl {
  using whole_kernel_registry = void *(*)(size_t, void *, size_t, scalar_t **);
  size_t inputs = 0, external_inputs = 0, outputs = 0;
  std::vector<size_t> input_pointer_offsets;
  std::vector<size_t> input_values;
  std::vector<size_t> entry_outputs;
  std::vector<size_t> output_entries;
  casadi::Function reference;
  std::vector<value_layout> values;
  std::vector<std::unique_ptr<operation>> operations;
  std::vector<size_t> panel_pointer_offsets;
  size_t value_panel_pointers = 0;
  std::vector<std::vector<const operation *>> entry_schedules;
  std::vector<size_t> factor_values;
  std::vector<bool> factor_spd;
  std::vector<size_t> spd_values;
  whole_kernel_registry whole_kernel = nullptr;
  struct direct_output { size_t value, panel, output, offset; };
  std::vector<direct_output> direct_outputs;
};

struct operation_entry_scope {
  casadi_mx_graph_plan_impl &plan;
  size_t begin;
  uint64_t entries;
  ~operation_entry_scope() {
    for (size_t i = begin; i < plan.operations.size(); ++i)
      plan.operations[i]->entries = entries;
  }
};

scalar_t execution_context::read(size_t value, index_t nz) const {
  if (nz < 0) return 0.;
  const auto &address = plan.values[value].csc[nz];
  if (address.panel == no_value) return 0.;
  return pointers(value)[address.panel][address.offset];
}

void execution_context::write(size_t value, size_t nz, scalar_t scalar) const {
  const auto &address = plan.values[value].csc[nz];
  if (address.panel == no_value)
    throw std::logic_error("cannot write an MX virtual zero");
  pointers(value)[address.panel][address.offset] = scalar;
}

void execution_context::zero(size_t value) const {
  const auto &layout = plan.values[value].matrix;
  auto &bound = pointers(value);
  for (size_t i = 0; i < layout.panels.size(); ++i) {
    const auto &panel = layout.panels[i];
    if (panel.pattern == sparsity::eye) continue;
    std::fill_n(bound[i], panel.pattern == sparsity::dense
                             ? panel.rows * panel.cols
                             : panel.rows,
                0.);
  }
}

struct backend_program_operation final : operation {
  panel_program_spec spec;
  panel_program_kernel kernel;
  backend_program_operation(panel_program_spec spec,
                            panel_program_kernel kernel)
      : spec(std::move(spec)), kernel(std::move(kernel)) {}
  void execute(execution_context &context) const override {
    kernel(context.instance.backend_pointers);
  }
};

struct input_operation final : operation {
  size_t output, input, offset = 0;
  bool direct = false;
  input_operation(size_t output, size_t input, bool direct)
      : output(output), input(input), direct(direct) {}
  void execute(execution_context &context) const override {
    const size_t begin = context.plan.input_pointer_offsets[input];
    const size_t end = context.plan.input_pointer_offsets[input + 1];
    if (direct) {
      auto &bound = context.pointers(output);
      bound.assign(context.external.begin() + begin,
                   context.external.begin() + end);
      return;
    }
    const scalar_t *source = context.external[begin];
    for (size_t nz = 0; nz < context.plan.values[output].csc.size(); ++nz)
      context.write(output, nz, source[nz]);
  }
  bool binding_only() const override { return direct; }
};

struct output_operation final : operation {
  size_t input, output, offset;
  bool direct = false;
  output_operation(size_t input, size_t output, size_t offset)
      : input(input), output(output), offset(offset) {}
  void execute(execution_context &context) const override {
    scalar_t *destination =
        context.external[context.plan.external_inputs + output] + offset;
    const auto &layout = context.plan.values[input];
    if (layout.direct_csc_panel() &&
        context.pointers(input)[0] == destination)
      return;
    for (size_t nz = 0; nz < layout.csc.size(); ++nz)
      destination[nz] = context.read(input, nz);
  }
  bool binding_only() const override { return direct; }
};

struct constant_operation final : operation {
  size_t output;
  std::vector<double> values;
  constant_operation(size_t output, std::vector<double> values)
      : output(output), values(std::move(values)) {}
  void execute(execution_context &context) const override {
    for (size_t i = 0; i < values.size(); ++i)
      context.write(output, i, values[i]);
  }
  bool binding_only() const override { return true; }
};

struct copy_segment {
  nz_address destination, source;
  ptrdiff_t destination_stride = 0, source_stride = 0;
  size_t count = 0;
  bool zero = false;
};

struct copy_operation final : operation {
  size_t input, output;
  std::vector<copy_segment> segments;
  double scale = 1.;
  index_t source_op = -1;
  copy_operation(size_t input, size_t output,
                 std::vector<copy_segment> segments,
                 double scale, index_t source_op)
      : input(input), output(output), segments(std::move(segments)),
        scale(scale), source_op(source_op) {}
  void execute(execution_context &context) const override {
    auto &destination = context.pointers(output);
    const auto &source = context.pointers(input);
    for (const auto &segment : segments) {
      scalar_t *dst = destination[segment.destination.panel] +
                      segment.destination.offset;
      if (segment.zero) {
        for (size_t i = 0; i < segment.count; ++i)
          dst[static_cast<ptrdiff_t>(i) * segment.destination_stride] = 0.;
        continue;
      }
      const scalar_t *src = source[segment.source.panel] +
                            segment.source.offset;
      for (size_t i = 0; i < segment.count; ++i)
        dst[static_cast<ptrdiff_t>(i) * segment.destination_stride] =
            scale * src[static_cast<ptrdiff_t>(i) * segment.source_stride];
    }
  }
};

struct transpose_operation final : operation {
  size_t input, output;
  transpose_operation(size_t input, size_t output)
      : input(input), output(output) {}
  void execute(execution_context &context) const override {
    const auto &input_layout = context.plan.values[input].matrix;
    const auto &output_layout = context.plan.values[output].matrix;
    const auto &source = context.pointers(input);
    auto &destination = context.pointers(output);
    for (size_t panel = 0; panel < input_layout.panels.size(); ++panel) {
      const auto &in = input_layout.panels[panel];
      const auto &out = output_layout.panels[panel];
      if (in.pattern == sparsity::eye) continue;
      if (in.pattern == sparsity::diag) {
        std::copy_n(source[panel], in.rows, destination[panel]);
        continue;
      }
      Eigen::Map<const matrix> input_view(source[panel], in.rows, in.cols);
      Eigen::Map<matrix> output_view(destination[panel], out.rows, out.cols);
      output_view.noalias() = input_view.transpose();
    }
  }
};

struct panel_alias {
  size_t output_panel = 0;
  size_t input_value = 0;
  size_t input_panel = 0;
  size_t input_offset = 0;
};

struct alias_operation final : operation {
  size_t output;
  std::vector<panel_alias> aliases;
  alias_operation(size_t output, std::vector<panel_alias> aliases)
      : output(output), aliases(std::move(aliases)) {}
  void execute(execution_context &context) const override {
    auto &destination = context.pointers(output);
    destination.resize(aliases.size());
    for (const auto &alias : aliases)
      destination[alias.output_panel] =
          context.pointers(alias.input_value)[alias.input_panel] +
          alias.input_offset;
  }
  bool binding_only() const override { return true; }
};

struct source_nz {
  size_t value = no_value;
  index_t nz = -1;
};

panel_alias canonical_panel_source(const casadi_mx_graph_plan_impl &plan,
                                   size_t value, size_t panel,
                                   size_t offset) {
  std::set<std::pair<size_t, size_t>> visited;
  while (visited.emplace(value, panel).second) {
    const alias_operation *producer = nullptr;
    for (const auto &operation : plan.operations) {
      const auto *alias = dynamic_cast<const alias_operation *>(operation.get());
      if (alias && alias->output == value) {
        producer = alias;
        break;
      }
    }
    if (!producer) break;
    const auto found = std::ranges::find_if(
        producer->aliases, [&](const panel_alias &alias) {
          return alias.output_panel == panel;
        });
    if (found == producer->aliases.end()) break;
    value = found->input_value;
    panel = found->input_panel;
    offset += found->input_offset;
  }
  return {0, value, panel, offset};
}

std::optional<std::vector<panel_alias>>
make_panel_aliases(const casadi_mx_graph_plan_impl &plan, size_t output,
                   std::span<const source_nz> source) {
  const auto &destination = plan.values[output];
  if (source.size() != destination.csc.size()) return std::nullopt;
  std::vector<std::vector<index_t>> panel_nz;
  panel_nz.reserve(destination.matrix.panels.size());
  for (const auto &panel : destination.matrix.panels) {
    if (panel.pattern == sparsity::eye) return std::nullopt;
    panel_nz.emplace_back(panel.pattern == sparsity::dense
                              ? panel.rows * panel.cols
                              : panel.rows,
                          -1);
  }
  for (size_t nz = 0; nz < destination.csc.size(); ++nz) {
    const auto &address = destination.csc[nz];
    if (address.panel == no_value) continue;
    if (address.panel >= panel_nz.size() ||
        address.offset >= panel_nz[address.panel].size())
      return std::nullopt;
    panel_nz[address.panel][address.offset] = static_cast<index_t>(nz);
  }
  std::vector<panel_alias> aliases;
  aliases.reserve(panel_nz.size());
  for (size_t panel = 0; panel < panel_nz.size(); ++panel) {
    if (panel_nz[panel].empty()) continue;
    const index_t first_nz = panel_nz[panel][0];
    if (first_nz < 0 || source[first_nz].value == no_value ||
        source[first_nz].nz < 0)
      return std::nullopt;
    const auto &first_address = plan.values[source[first_nz].value]
                                    .csc[source[first_nz].nz];
    if (first_address.panel == no_value) return std::nullopt;
    const auto first = canonical_panel_source(
        plan, source[first_nz].value, first_address.panel,
        first_address.offset);
    for (size_t offset = 0; offset < panel_nz[panel].size(); ++offset) {
      const index_t output_nz = panel_nz[panel][offset];
      if (output_nz < 0 || source[output_nz].value != source[first_nz].value ||
          source[output_nz].nz < 0)
        return std::nullopt;
      const auto &input_address = plan.values[source[output_nz].value]
                                      .csc[source[output_nz].nz];
      if (input_address.panel == no_value) return std::nullopt;
      const auto input = canonical_panel_source(
          plan, source[output_nz].value, input_address.panel,
          input_address.offset);
      if (input.input_value != first.input_value ||
          input.input_panel != first.input_panel ||
          input.input_offset != first.input_offset + offset)
        return std::nullopt;
    }
    aliases.push_back(
        {panel, first.input_value, first.input_panel, first.input_offset});
  }
  return aliases;
}

bool append_alias_or_copy(casadi_mx_graph_plan_impl &plan, size_t input,
                          size_t output, std::vector<index_t> map,
                          double scale = 1., index_t source_op = -1) {
  if (scale == 1.) {
    std::vector<source_nz> source;
    source.reserve(map.size());
    for (const index_t nz : map) source.push_back({input, nz});
    if (auto aliases = make_panel_aliases(plan, output, source)) {
      plan.values[output].allocate = false;
      plan.operations.push_back(std::make_unique<alias_operation>(
          output, std::move(*aliases)));
      return true;
    }
  }
  std::vector<copy_segment> segments;
  const auto &source = plan.values[input].csc;
  const auto &destination = plan.values[output].csc;
  for (size_t begin = 0; begin < map.size();) {
    const bool source_zero =
        map[begin] < 0 || source[map[begin]].panel == no_value;
    copy_segment segment{
        .destination = destination[begin],
        .source = source_zero ? nz_address{} : source[map[begin]],
        .count = 1,
        .zero = source_zero};
    if (begin + 1 < map.size()) {
      const auto &next_destination = destination[begin + 1];
      if (next_destination.panel == segment.destination.panel)
        segment.destination_stride =
            static_cast<ptrdiff_t>(next_destination.offset) -
            static_cast<ptrdiff_t>(segment.destination.offset);
      if (segment.zero) {
        if (map[begin + 1] >= 0 &&
            source[map[begin + 1]].panel != no_value)
          segment.destination_stride = 0;
      } else if (map[begin + 1] >= 0) {
        const auto &next_source = source[map[begin + 1]];
        if (next_source.panel == segment.source.panel)
          segment.source_stride =
              static_cast<ptrdiff_t>(next_source.offset) -
              static_cast<ptrdiff_t>(segment.source.offset);
      }
    }
    while (segment.destination_stride && begin + segment.count < map.size()) {
      const size_t index = begin + segment.count;
      const auto &next_destination = destination[index];
      if (next_destination.panel != segment.destination.panel ||
          static_cast<ptrdiff_t>(next_destination.offset) !=
              static_cast<ptrdiff_t>(segment.destination.offset) +
                  segment.destination_stride *
                      static_cast<ptrdiff_t>(segment.count))
        break;
      if (segment.zero) {
        if (map[index] >= 0 && source[map[index]].panel != no_value) break;
      } else {
        if (map[index] < 0 || source[map[index]].panel == no_value) break;
        const auto &next_source = source[map[index]];
        if (next_source.panel != segment.source.panel ||
            static_cast<ptrdiff_t>(next_source.offset) !=
                static_cast<ptrdiff_t>(segment.source.offset) +
                    segment.source_stride *
                        static_cast<ptrdiff_t>(segment.count))
          break;
      }
      ++segment.count;
    }
    segments.push_back(segment);
    begin += segment.count;
  }
  plan.operations.push_back(std::make_unique<copy_operation>(
      input, output, std::move(segments), scale, source_op));
  return false;
}

std::unique_ptr<copy_operation>
make_dense_materialization(const casadi_mx_graph_plan_impl &plan,
                           size_t input, size_t output) {
  const auto &source = plan.values[input];
  const auto &destination = plan.values[output];
  if (!destination.sparsity.is_dense() ||
      !destination.direct_csc_panel())
    throw std::logic_error("dense graph materialization needs dense storage");

  std::vector<copy_segment> segments;
  segments.push_back({.destination = {0, 0},
                      .destination_stride = 1,
                      .count = destination.csc.size(),
                      .zero = true});
  std::vector<std::vector<std::pair<nz_address, nz_address>>> by_panel(
      source.matrix.panels.size());
  const auto map = coordinate_map(destination.sparsity, source.sparsity);
  for (size_t nz = 0; nz < map.size(); ++nz) {
    if (map[nz] < 0) continue;
    const auto address = source.csc[map[nz]];
    if (address.panel == no_value) continue;
    by_panel[address.panel].emplace_back(destination.csc[nz], address);
  }
  for (const auto &entries : by_panel) {
    for (size_t begin = 0; begin < entries.size();) {
      copy_segment segment{.destination = entries[begin].first,
                           .source = entries[begin].second,
                           .count = 1};
      if (begin + 1 < entries.size()) {
        segment.destination_stride =
            static_cast<ptrdiff_t>(entries[begin + 1].first.offset) -
            static_cast<ptrdiff_t>(segment.destination.offset);
        segment.source_stride =
            static_cast<ptrdiff_t>(entries[begin + 1].second.offset) -
            static_cast<ptrdiff_t>(segment.source.offset);
      }
      while (segment.destination_stride &&
             begin + segment.count < entries.size()) {
        const auto &[next_destination, next_source] =
            entries[begin + segment.count];
        if (static_cast<ptrdiff_t>(next_destination.offset) !=
                static_cast<ptrdiff_t>(segment.destination.offset) +
                    segment.destination_stride *
                        static_cast<ptrdiff_t>(segment.count) ||
            static_cast<ptrdiff_t>(next_source.offset) !=
                static_cast<ptrdiff_t>(segment.source.offset) +
                    segment.source_stride *
                        static_cast<ptrdiff_t>(segment.count))
          break;
        ++segment.count;
      }
      segments.push_back(segment);
      begin += segment.count;
    }
  }
  return std::make_unique<copy_operation>(input, output, std::move(segments),
                                          1., -1);
}

struct binary_operation final : operation {
  size_t lhs, rhs, output;
  index_t op;
  std::vector<index_t> lhs_map, rhs_map;
  binary_operation(size_t lhs, size_t rhs, size_t output, index_t op,
                   std::vector<index_t> lhs_map,
                   std::vector<index_t> rhs_map)
      : lhs(lhs), rhs(rhs), output(output), op(op),
        lhs_map(std::move(lhs_map)), rhs_map(std::move(rhs_map)) {}
  void execute(execution_context &context) const override {
    for (size_t i = 0; i < lhs_map.size(); ++i) {
      const double a = context.read(lhs, lhs_map[i]);
      const double b = context.read(rhs, rhs_map[i]);
      const double value = op == casadi::OP_ADD ? a + b
                           : op == casadi::OP_SUB ? a - b
                           : op == casadi::OP_MUL ? a * b
                                                  : a / b;
      context.write(output, i, value);
    }
  }
};

struct scatter_operation final : operation {
  size_t base, source, output;
  std::vector<index_t> base_map, destination;
  bool add = false, scalar = false;
  scatter_operation(size_t base, size_t source, size_t output,
                    std::vector<index_t> base_map,
                    std::vector<index_t> destination, bool add, bool scalar)
      : base(base), source(source), output(output),
        base_map(std::move(base_map)), destination(std::move(destination)),
        add(add), scalar(scalar) {}
  void execute(execution_context &context) const override {
    for (size_t i = 0; i < base_map.size(); ++i)
      context.write(output, i, context.read(base, base_map[i]));
    for (size_t i = 0; i < destination.size(); ++i) {
      if (destination[i] < 0) continue;
      const double value = context.read(source, scalar ? 0 : i);
      if (add) context.write(output, destination[i],
                             context.read(output, destination[i]) + value);
      else context.write(output, destination[i], value);
    }
  }
};

struct concat_operation final : operation {
  size_t output;
  std::vector<size_t> inputs;
  std::vector<std::vector<index_t>> destinations;
  concat_operation(size_t output, std::vector<size_t> inputs,
                   std::vector<std::vector<index_t>> destinations)
      : output(output), inputs(std::move(inputs)),
        destinations(std::move(destinations)) {}
  void execute(execution_context &context) const override {
    for (size_t dep = 0; dep < inputs.size(); ++dep)
      for (size_t nz = 0; nz < destinations[dep].size(); ++nz)
        context.write(output, destinations[dep][nz],
                      context.read(inputs[dep], nz));
  }
};

struct product_operation final : operation {
  size_t addend, lhs, rhs, output;
  batch_product_kernel kernel;
  std::vector<index_t> addend_map;
  double addend_scale = 1., product_scale = 1.;
  product_operation(size_t addend, size_t lhs, size_t rhs, size_t output,
                    batch_product_kernel kernel,
                    std::vector<index_t> addend_map)
      : addend(addend), lhs(lhs), rhs(rhs), output(output),
        kernel(std::move(kernel)), addend_map(std::move(addend_map)) {}
  void execute(execution_context &context) const override {
    kernel(context.instance.backend_pointers);
  }
};

struct solve_operation final : operation {
  struct segment {
    size_t rhs_panel = 0, output_panel = 0;
    size_t rhs_offset = 0, output_offset = 0;
    size_t rhs_leading = 0, output_leading = 0;
    size_t cols = 0;
  };
  size_t rhs = no_value, matrix_value = no_value, output = no_value;
  size_t factor_slot = no_value;
  size_t workspace_slot = no_value;
  std::vector<index_t> active_columns;
  std::vector<segment> segments;
  bool transpose = false, identity_rhs = false, spd = false;
  solve_operation(size_t rhs, size_t matrix, size_t output, bool transpose,
                  bool identity_rhs)
      : rhs(rhs), matrix_value(matrix), output(output),
        transpose(transpose), identity_rhs(identity_rhs) {}
  void execute(execution_context &context) const override {
    const auto &matrix_layout = context.plan.values[matrix_value];
    const size_t n = matrix_layout.matrix.rows;
    const auto &rhs_layout = context.plan.values[rhs];
    const sparse_index bi(rhs_layout.sparsity);
    if (active_columns.empty()) {
      context.zero(output);
      return;
    }
    const size_t cols = active_columns.size();
    auto &cache = *context.instance.factors.at(factor_slot);
    if (cache.epoch != context.instance.factor_epoch) {
      if (matrix_layout.direct_csc_panel()) {
        Eigen::Map<matrix> a(context.pointers(matrix_value)[0], n, n);
        cache.compute(a, spd);
      } else {
        matrix a = matrix::Zero(n, n);
        const sparse_index ai(matrix_layout.sparsity);
        for (size_t col = 0; col < n; ++col)
          for (index_t nz = ai.colind[col]; nz < ai.colind[col + 1]; ++nz)
            a(ai.row[nz], col) = context.read(matrix_value, nz);
        cache.compute(a, spd);
      }
      cache.epoch = context.instance.factor_epoch;
    }
    const auto &output_layout = context.plan.values[output];
    if (identity_rhs && output_layout.direct_csc_panel()) {
      Eigen::Map<matrix> x(context.pointers(output)[0], n, n);
      cache.inverse(x);
      return;
    }
    if (rhs_layout.direct_csc_panel() &&
        output_layout.direct_csc_panel() &&
        active_columns.size() == static_cast<size_t>(bi.cols)) {
      Eigen::Map<const matrix> b(context.pointers(rhs)[0], n, cols);
      Eigen::Map<matrix> x(context.pointers(output)[0], n, cols);
      if (transpose)
        cache.transpose_solve(b, x);
      else
        cache.solve(b, x);
      return;
    }
    if (!segments.empty()) {
      using const_view = Eigen::Map<const matrix, Eigen::Unaligned,
                                    Eigen::OuterStride<>>;
      using view = Eigen::Map<matrix, Eigen::Unaligned,
                              Eigen::OuterStride<>>;
      for (const auto &segment : segments) {
        const_view b(context.pointers(rhs)[segment.rhs_panel] +
                         segment.rhs_offset,
                     n, segment.cols,
                     Eigen::OuterStride<>(segment.rhs_leading));
        view x(context.pointers(output)[segment.output_panel] +
                   segment.output_offset,
               n, segment.cols,
               Eigen::OuterStride<>(segment.output_leading));
        if (transpose)
          cache.transpose_solve(b, x);
        else
          cache.solve(b, x);
      }
      return;
    }
    matrix &b = context.instance.solve_rhs_buffers[workspace_slot];
    b.resize(n, cols);
    b.setZero();
    for (size_t packed = 0; packed < active_columns.size(); ++packed) {
      const index_t col = active_columns[packed];
      for (index_t nz = bi.colind[col]; nz < bi.colind[col + 1]; ++nz)
        b(bi.row[nz], packed) = context.read(rhs, nz);
    }
    if (output_layout.direct_csc_panel() &&
        active_columns.size() == static_cast<size_t>(bi.cols)) {
      Eigen::Map<matrix> x(context.pointers(output)[0], n, cols);
      if (transpose)
        cache.transpose_solve(b, x);
      else
        cache.solve(b, x);
      return;
    }
    context.zero(output);
    matrix &x = context.instance.solve_output_buffers[workspace_slot];
    x.resize(n, cols);
    if (transpose)
      cache.transpose_solve(b, x);
    else
      cache.solve(b, x);
    const sparse_index oi(output_layout.sparsity);
    for (size_t packed = 0; packed < active_columns.size(); ++packed) {
      const index_t col = active_columns[packed];
      for (index_t nz = oi.colind[col]; nz < oi.colind[col + 1]; ++nz)
        context.write(output, nz, x(oi.row[nz], packed));
    }
  }
};

panel_program_operand panel_operand(const casadi_mx_graph_plan_impl &plan,
                                    size_t value, index_t nz) {
  if (nz < 0) return {};
  const auto &address = plan.values[value].csc.at(nz);
  if (address.panel == no_value) return {};
  return {plan.panel_pointer_offsets.at(value) + address.panel,
          address.offset, 0};
}

panel_program_operand panel_operand(const casadi_mx_graph_plan_impl &plan,
                                    size_t value, const nz_address &address,
                                    ptrdiff_t stride = 0) {
  if (address.panel == no_value) return {};
  return {plan.panel_pointer_offsets.at(value) + address.panel,
          address.offset, stride};
}

panel_program_operand external_operand(
    const casadi_mx_graph_plan_impl &plan, size_t pointer,
    size_t offset = 0, ptrdiff_t stride = 0) {
  return {plan.value_panel_pointers + pointer, offset, stride};
}

void append_assignment(panel_program_spec &program,
                       panel_program_operand destination,
                       panel_program_operand source, double scale = 1.) {
  if (source.pointer == panel_program_operand::invalid)
    program.instructions.push_back(
        {.op = panel_program_op::fill,
         .destination = destination,
         .scalar = 0.});
  else
    program.instructions.push_back(
        {.op = panel_program_op::copy,
         .destination = destination,
         .lhs = source,
         .scalar = scale});
}

bool lower_to_panel_program(const casadi_mx_graph_plan_impl &plan,
                            const operation &operation,
                            panel_program_spec &program) {
  if (const auto *input = dynamic_cast<const input_operation *>(&operation)) {
    if (input->direct) return false;
    const size_t external = plan.input_pointer_offsets.at(input->input);
    for (size_t nz = 0; nz < plan.values[input->output].csc.size(); ++nz)
      append_assignment(program, panel_operand(plan, input->output, nz),
                        external_operand(plan, external, nz));
    return true;
  }
  if (const auto *output =
          dynamic_cast<const output_operation *>(&operation)) {
    for (size_t nz = 0; nz < plan.values[output->input].csc.size(); ++nz)
      append_assignment(
          program,
          external_operand(plan, plan.external_inputs + output->output,
                           output->offset + nz),
          panel_operand(plan, output->input, nz));
    return true;
  }
  if (const auto *copy = dynamic_cast<const copy_operation *>(&operation)) {
    for (const auto &segment : copy->segments) {
      auto destination = panel_operand(
          plan, copy->output, segment.destination,
          segment.destination_stride);
      if (segment.zero) {
        program.instructions.push_back(
            {.op = panel_program_op::fill,
             .destination = destination,
             .count = segment.count,
             .scalar = 0.});
      } else {
        program.instructions.push_back(
            {.op = panel_program_op::copy,
             .destination = destination,
             .lhs = panel_operand(plan, copy->input, segment.source,
                                  segment.source_stride),
             .count = segment.count,
             .scalar = copy->scale});
      }
    }
    return true;
  }
  if (const auto *binary =
          dynamic_cast<const binary_operation *>(&operation)) {
    for (size_t nz = 0; nz < binary->lhs_map.size(); ++nz) {
      const auto destination = panel_operand(plan, binary->output, nz);
      const auto lhs = panel_operand(plan, binary->lhs, binary->lhs_map[nz]);
      const auto rhs = panel_operand(plan, binary->rhs, binary->rhs_map[nz]);
      const bool has_lhs = lhs.pointer != panel_program_operand::invalid;
      const bool has_rhs = rhs.pointer != panel_program_operand::invalid;
      if (!has_lhs || !has_rhs) {
        if (binary->op == casadi::OP_ADD)
          append_assignment(program, destination, has_lhs ? lhs : rhs);
        else if (binary->op == casadi::OP_SUB)
          append_assignment(program, destination, has_lhs ? lhs : rhs,
                            has_lhs ? 1. : -1.);
        else
          append_assignment(program, destination, {}, 0.);
        continue;
      }
      program.instructions.push_back(
          {.op = binary->op == casadi::OP_ADD ? panel_program_op::add
                 : binary->op == casadi::OP_SUB ? panel_program_op::sub
                 : binary->op == casadi::OP_MUL ? panel_program_op::mul
                                                 : panel_program_op::div,
           .destination = destination,
           .lhs = lhs,
           .rhs = rhs});
    }
    return true;
  }
  if (const auto *scatter =
          dynamic_cast<const scatter_operation *>(&operation)) {
    for (size_t nz = 0; nz < scatter->base_map.size(); ++nz)
      append_assignment(program, panel_operand(plan, scatter->output, nz),
                        panel_operand(plan, scatter->base,
                                      scatter->base_map[nz]));
    for (size_t nz = 0; nz < scatter->destination.size(); ++nz) {
      if (scatter->destination[nz] < 0) continue;
      const auto destination = panel_operand(
          plan, scatter->output, scatter->destination[nz]);
      const auto source = panel_operand(
          plan, scatter->source, scatter->scalar ? 0 : nz);
      if (scatter->add)
        program.instructions.push_back(
            {.op = panel_program_op::add,
             .destination = destination,
             .lhs = destination,
             .rhs = source});
      else
        append_assignment(program, destination, source);
    }
    return true;
  }
  if (const auto *concat =
          dynamic_cast<const concat_operation *>(&operation)) {
    for (size_t dependency = 0; dependency < concat->inputs.size();
         ++dependency)
      for (size_t nz = 0; nz < concat->destinations[dependency].size(); ++nz)
        append_assignment(
            program,
            panel_operand(plan, concat->output,
                          concat->destinations[dependency][nz]),
            panel_operand(plan, concat->inputs[dependency], nz));
    return true;
  }
  return false;
}

size_t operation_uses_value(const operation &operation, size_t value) {
  size_t uses = 0;
  const auto count = [&](size_t input) {
    if (input == value) ++uses;
  };
  if (const auto *output = dynamic_cast<const output_operation *>(&operation))
    count(output->input);
  else if (const auto *copy =
               dynamic_cast<const copy_operation *>(&operation))
    count(copy->input);
  else if (const auto *transpose =
               dynamic_cast<const transpose_operation *>(&operation))
    count(transpose->input);
  else if (const auto *alias =
               dynamic_cast<const alias_operation *>(&operation))
    for (const auto &entry : alias->aliases) count(entry.input_value);
  else if (const auto *binary =
               dynamic_cast<const binary_operation *>(&operation)) {
    count(binary->lhs);
    count(binary->rhs);
  } else if (const auto *scatter =
                 dynamic_cast<const scatter_operation *>(&operation)) {
    count(scatter->base);
    count(scatter->source);
  } else if (const auto *concat =
                 dynamic_cast<const concat_operation *>(&operation))
    for (const size_t input : concat->inputs) count(input);
  else if (const auto *product =
               dynamic_cast<const product_operation *>(&operation)) {
    count(product->addend);
    count(product->lhs);
    count(product->rhs);
  } else if (const auto *solve =
                 dynamic_cast<const solve_operation *>(&operation)) {
    count(solve->rhs);
    count(solve->matrix_value);
  }
  return uses;
}

bool zero_product_addend(const casadi_mx_graph_plan_impl &plan,
                         const product_operation &product) {
  if (std::ranges::all_of(product.addend_map,
                          [](index_t nz) { return nz < 0; }))
    return true;
  const auto found = std::ranges::find_if(
      plan.operations, [&](const auto &operation) {
        const auto *constant =
            dynamic_cast<const constant_operation *>(operation.get());
        return constant && constant->output == product.addend &&
               std::ranges::all_of(constant->values,
                                   [](double value) { return value == 0.; });
      });
  return found != plan.operations.end();
}

void lazily_apply_inverses(casadi_mx_graph_plan_impl &plan) {
  struct inverse_source {
    solve_operation *solve = nullptr;
    bool transpose = false;
  };
  std::unordered_map<size_t, inverse_source> inverses;
  for (const auto &operation : plan.operations)
    if (auto *solve = dynamic_cast<solve_operation *>(operation.get());
        solve && solve->identity_rhs)
      inverses.emplace(solve->output, inverse_source{solve, false});
  std::set<size_t> inverse_aliases;
  for (const auto &operation : plan.operations) {
    const auto *alias = dynamic_cast<const alias_operation *>(operation.get());
    if (!alias || alias->aliases.empty()) continue;
    const size_t input = alias->aliases.front().input_value;
    const auto found = inverses.find(input);
    if (found == inverses.end()) continue;
    const auto &source = plan.values[input].matrix;
    const auto &destination = plan.values[alias->output].matrix;
    if (source.rows != destination.cols ||
        source.cols != destination.rows ||
        !std::ranges::all_of(alias->aliases, [&](const panel_alias &panel) {
          if (panel.input_value != input || panel.input_offset ||
              panel.output_panel >= destination.panels.size() ||
              panel.input_panel >= source.panels.size())
            return false;
          const auto &in = source.panels[panel.input_panel];
          const auto &out = destination.panels[panel.output_panel];
          return out.row_offset == in.col_offset &&
                 out.col_offset == in.row_offset && out.rows == in.cols &&
                 out.cols == in.rows && out.transposed != in.transposed;
        }))
      continue;
    inverses.emplace(alias->output,
                     inverse_source{found->second.solve, true});
    inverse_aliases.insert(alias->output);
  }

  for (auto &operation : plan.operations) {
    auto *product = dynamic_cast<product_operation *>(operation.get());
    if (!product || product->product_scale != 1. ||
        !zero_product_addend(plan, *product))
      continue;
    const auto found = inverses.find(product->lhs);
    if (found == inverses.end()) continue;
    auto replacement = std::make_unique<solve_operation>(
        product->rhs, found->second.solve->matrix_value, product->output,
        found->second.transpose, false);
    replacement->entries = product->entries;
    operation = std::move(replacement);
  }

  std::erase_if(plan.operations, [&](const auto &operation) {
    const auto *alias = dynamic_cast<const alias_operation *>(operation.get());
    if (!alias || !inverse_aliases.contains(alias->output)) return false;
    for (const auto &consumer : plan.operations)
      if (consumer.get() != operation.get() &&
          operation_uses_value(*consumer, alias->output))
        return false;
    plan.values[alias->output].allocate = false;
    return true;
  });
  std::erase_if(plan.operations, [&](const auto &operation) {
    const auto *solve = dynamic_cast<const solve_operation *>(operation.get());
    if (!solve || !solve->identity_rhs) return false;
    for (const auto &consumer : plan.operations)
      if (consumer.get() != operation.get() &&
          operation_uses_value(*consumer, solve->output))
        return false;
    plan.values[solve->output].allocate = false;
    return true;
  });
}

void fuse_lazy_products(casadi_mx_graph_plan_impl &plan) {
  std::unordered_map<size_t, product_operation *> products;
  std::unordered_map<const operation *, size_t> original_position;
  for (size_t i = 0; i < plan.operations.size(); ++i)
    original_position.emplace(plan.operations[i].get(), i);
  for (const auto &operation : plan.operations)
    if (auto *product = dynamic_cast<product_operation *>(operation.get()))
      products.emplace(product->output, product);
  std::unordered_map<size_t, size_t> uses;
  for (const auto &[value, product] : products) {
    (void)product;
    size_t count = 0;
    for (const auto &operation : plan.operations)
      count += operation_uses_value(*operation, value);
    uses.emplace(value, count);
  }
  std::set<const operation *> remove;
  std::unordered_map<const operation *, size_t> schedule_at;
  for (size_t operation_index = 0; operation_index < plan.operations.size();
       ++operation_index) {
    const auto &operation = plan.operations[operation_index];
    auto *binary = dynamic_cast<binary_operation *>(operation.get());
    if (!binary || (binary->op != casadi::OP_ADD &&
                    binary->op != casadi::OP_SUB))
      continue;
    const auto try_fuse = [&](bool product_is_lhs) {
      const size_t product_value = product_is_lhs ? binary->lhs : binary->rhs;
      const auto found = products.find(product_value);
      if (found == products.end() || uses.at(product_value) != 1)
        return false;
      auto &product = *found->second;
      if (product.entries != binary->entries ||
          !zero_product_addend(plan, product))
        return false;
      const auto &product_map =
          product_is_lhs ? binary->lhs_map : binary->rhs_map;
      if (product_map.size() != plan.values[binary->output].csc.size())
        return false;
      for (size_t nz = 0; nz < product_map.size(); ++nz)
        if (product_map[nz] != static_cast<index_t>(nz)) return false;
      const size_t old_output = product.output;
      product.output = binary->output;
      product.addend = product_is_lhs ? binary->rhs : binary->lhs;
      product.addend_map =
          product_is_lhs ? binary->rhs_map : binary->lhs_map;
      product.product_scale =
          binary->op == casadi::OP_SUB && !product_is_lhs ? -1. : 1.;
      product.addend_scale =
          binary->op == casadi::OP_SUB && product_is_lhs ? -1. : 1.;
      ensure_product_coverage(plan.values[product.output],
                              plan.values[product.lhs],
                              plan.values[product.rhs]);
      plan.values[old_output].allocate = false;
      // The addend may be produced between the original product and this
      // binary node.  The fused C +/- A*B operation therefore belongs at the
      // binary node's topological position, not at the old product position.
      schedule_at[&product] = operation_index;
      remove.insert(binary);
      return true;
    };
    if (!try_fuse(true)) try_fuse(false);
  }
  std::erase_if(plan.operations, [&](const auto &operation) {
    return remove.contains(operation.get());
  });
  std::stable_sort(plan.operations.begin(), plan.operations.end(),
                   [&](const auto &lhs, const auto &rhs) {
    const auto position = [&](const auto &value) {
      if (const auto found = schedule_at.find(value.get());
          found != schedule_at.end())
        return found->second;
      return original_position.at(value.get());
    };
    return position(lhs) < position(rhs);
  });
}

bool identity_map(std::span<const index_t> map) {
  for (size_t i = 0; i < map.size(); ++i)
    if (map[i] != static_cast<index_t>(i)) return false;
  return true;
}

void fuse_product_accumulations(casadi_mx_graph_plan_impl &plan) {
  std::unordered_map<size_t, product_operation *> producers;
  for (const auto &operation : plan.operations)
    if (auto *product = dynamic_cast<product_operation *>(operation.get()))
      producers.emplace(product->output, product);

  std::unordered_map<size_t, size_t> uses;
  for (const auto &[value, producer] : producers) {
    (void)producer;
    for (const auto &operation : plan.operations)
      uses[value] += operation_uses_value(*operation, value);
  }

  for (auto operation = plan.operations.rbegin();
       operation != plan.operations.rend(); ++operation) {
    auto *product = dynamic_cast<product_operation *>(operation->get());
    if (!product || product->addend_scale != 1. ||
        !identity_map(product->addend_map))
      continue;
    const auto found = producers.find(product->addend);
    if (found == producers.end() || uses[product->addend] != 1) continue;
    auto &predecessor = *found->second;
    if (predecessor.entries != product->entries ||
        plan.values[predecessor.output].sparsity !=
            plan.values[product->output].sparsity)
      continue;

    const size_t intermediate = predecessor.output;
    predecessor.output = product->output;
    ensure_product_coverage(plan.values[predecessor.output],
                            plan.values[predecessor.lhs],
                            plan.values[predecessor.rhs]);
    product->addend = product->output;
    plan.values[intermediate].allocate = false;
  }
}

void reuse_dense_output_materializations(casadi_mx_graph_plan_impl &plan) {
  struct candidate {
    size_t source = no_value;
    size_t dense = no_value;
    size_t alias_operation = no_value;
  };
  std::vector<candidate> candidates;
  std::set<size_t> claimed_sources;
  for (const auto &operation : plan.operations) {
    const auto *output = dynamic_cast<const output_operation *>(operation.get());
    if (!output) continue;
    const size_t dense = output->input;
    if (!plan.values[dense].sparsity.is_dense() ||
        plan.values[dense].direct_csc_panel())
      continue;
    for (size_t producer_index = 0; producer_index < plan.operations.size();
         ++producer_index) {
      const auto *alias = dynamic_cast<const alias_operation *>(
          plan.operations[producer_index].get());
      if (!alias || alias->output != dense || alias->aliases.empty())
        continue;
      const size_t source = alias->aliases.front().input_value;
      if (!std::ranges::all_of(
              alias->aliases, [&](const panel_alias &entry) {
                return entry.input_value == source;
              }) ||
          std::ranges::any_of(
              plan.values[source].matrix.panels,
              [](const panel_layout &panel) {
                return panel.pattern != sparsity::dense;
              }) ||
          plan.values[source].matrix.rows != plan.values[dense].matrix.rows ||
          plan.values[source].matrix.cols != plan.values[dense].matrix.cols ||
          !claimed_sources.insert(source).second)
        break;
      size_t product_uses = 0;
      for (size_t i = producer_index + 1; i < plan.operations.size(); ++i)
        if (const auto *product = dynamic_cast<const product_operation *>(
                plan.operations[i].get())) {
          product_uses += product->lhs == source;
          product_uses += product->rhs == source;
        }
      if (std::getenv("MOTO_TRACE_GRAPH_LOWERING"))
        std::cerr << "dense graph output " << dense << " aliases " << source
                  << " and has " << product_uses
                  << " downstream product uses\n";
      if (product_uses > 1)
        candidates.push_back({source, dense, producer_index});
      else
        claimed_sources.erase(source);
      break;
    }
  }

  for (const auto &item : candidates) {
    const std::string fingerprint = plan.values[item.dense].fingerprint;
    const uint64_t entries = plan.operations[item.alias_operation]->entries;
    plan.values[item.dense] = make_layout(plan.values[item.dense].sparsity);
    plan.values[item.dense].fingerprint = fingerprint;
    auto materialize =
        make_dense_materialization(plan, item.source, item.dense);
    materialize->entries = entries;
    plan.operations[item.alias_operation] = std::move(materialize);
    for (size_t i = item.alias_operation + 1; i < plan.operations.size(); ++i)
      if (auto *product =
              dynamic_cast<product_operation *>(plan.operations[i].get())) {
        if (product->lhs == item.source) product->lhs = item.dense;
        if (product->rhs == item.source) product->rhs = item.dense;
      }
  }
}

void materialize_reused_product_branches(casadi_mx_graph_plan_impl &plan) {
  for (;;) {
    struct choice {
      size_t value = no_value;
      size_t first_use = no_value;
      size_t uses = 0;
      size_t saved_pairs = 0;
      uint64_t entries = 0;
    } best;
    for (size_t value = 0; value < plan.values.size(); ++value) {
      const auto &value_layout = plan.values[value];
      const auto &layout = value_layout.matrix;
      // A dense composite view is still lazy storage.  Once it branches to
      // several products, materialize that SSA value once so every consumer
      // shares the same computation.  Sparse views retain their panel layout.
      if (value_layout.allocate || !value_layout.sparsity.is_dense() ||
          layout.panels.size() < 2 || !layout.rows || !layout.cols)
        continue;
      if (std::ranges::any_of(layout.panels, [](const panel_layout &panel) {
            return panel.pattern != sparsity::dense;
          }))
        continue;
      size_t uses = 0;
      size_t first_use = no_value;
      size_t fragmented_pairs = 0;
      size_t compact_pairs = 0;
      uint64_t entries = 0;
      bool compatible_entries = true;
      for (size_t i = 0; i < plan.operations.size(); ++i) {
        const auto *product = dynamic_cast<const product_operation *>(
            plan.operations[i].get());
        if (!product || (!(product->entries & uint64_t{1}))) continue;
        const bool lhs = product->lhs == value;
        const bool rhs = product->rhs == value;
        if (!lhs && !rhs) continue;
        if (!entries) entries = product->entries;
        else compatible_entries &= entries == product->entries;
        first_use = std::min(first_use, i);
        uses += lhs + rhs;
        const auto &left = plan.values[product->lhs].matrix;
        const auto &right = plan.values[product->rhs].matrix;
        for (const auto &l : left.panels)
          for (const auto &r : right.panels)
            fragmented_pairs +=
                std::max(l.col_offset, r.row_offset) <
                std::min(l.col_offset + l.cols, r.row_offset + r.rows);
        compact_pairs += lhs ? right.panels.size() : left.panels.size();
      }
      const size_t saved_pairs = fragmented_pairs > compact_pairs
                                     ? fragmented_pairs - compact_pairs
                                     : 0;
      const bool fragmented = compact_pairs &&
          fragmented_pairs >= 4 * compact_pairs;
      if (compatible_entries && uses && (uses > 1 || fragmented) &&
          (uses > best.uses ||
           (uses == best.uses && saved_pairs > best.saved_pairs)))
        best = {value, first_use, uses, saved_pairs, entries};
    }
    if (best.value == no_value) return;

    const size_t rows = plan.values[best.value].matrix.rows;
    const size_t cols = plan.values[best.value].matrix.cols;
    const casadi::Sparsity dense = casadi::Sparsity::dense(
        static_cast<casadi_int>(rows), static_cast<casadi_int>(cols));
    const size_t materialized = plan.values.size();
    plan.values.push_back(make_layout(dense));
    plan.values.back().fingerprint =
        plan.values[best.value].fingerprint + ":consumer_dense";
    auto copy = make_dense_materialization(plan, best.value, materialized);
    copy->entries = best.entries;
    plan.values[materialized].local_entries = best.entries;
    for (auto &operation : plan.operations)
      if (auto *product = dynamic_cast<product_operation *>(operation.get());
          product && product->entries == best.entries) {
        if (product->lhs == best.value) product->lhs = materialized;
        if (product->rhs == best.value) product->rhs = materialized;
      }
    plan.operations.insert(plan.operations.begin() + best.first_use,
                           std::move(copy));
  }
}

void materialize_solve_inputs(casadi_mx_graph_plan_impl &plan) {
  const size_t original_operations = plan.operations.size();
  std::map<size_t, size_t> dense_value;
  std::map<size_t, size_t> dense_rhs;
  std::map<size_t, uint64_t> entries;
  std::map<size_t, uint64_t> rhs_entries;
  for (const auto &operation : plan.operations) {
    const auto *solve = dynamic_cast<const solve_operation *>(operation.get());
    if (!solve) continue;
    if (!plan.values[solve->matrix_value].direct_csc_panel())
      entries[solve->matrix_value] |= solve->entries;
    const auto &rhs = plan.values[solve->rhs];
    const sparse_index rhs_index(rhs.sparsity);
    bool all_columns_active = true;
    for (index_t col = 0; col < rhs_index.cols; ++col)
      if (rhs_index.colind[col] == rhs_index.colind[col + 1]) {
        all_columns_active = false;
        break;
      }
    if (!rhs.direct_csc_panel() && all_columns_active)
      rhs_entries[solve->rhs] |= solve->entries;
  }
  std::map<size_t, std::unique_ptr<operation>> materializations;
  std::map<size_t, std::unique_ptr<operation>> rhs_materializations;
  const auto materialize_dense = [&](size_t source) {
    const size_t rows = plan.values[source].matrix.rows;
    const size_t cols = plan.values[source].matrix.cols;
    const std::string fingerprint = plan.values[source].fingerprint;
    const casadi::Sparsity dense = casadi::Sparsity::dense(
        static_cast<casadi_int>(rows), static_cast<casadi_int>(cols));
    const size_t output = plan.values.size();
    plan.values.push_back(make_layout(dense));
    plan.values.back().fingerprint = fingerprint;
    auto materialize = make_dense_materialization(plan, source, output);
    return std::pair{output, std::move(materialize)};
  };
  for (const auto &[source, source_entries] : entries) {
    const auto &layout = plan.values[source];
    if (layout.matrix.rows != layout.matrix.cols)
      throw std::runtime_error("graph factor input must be square");
    auto [output, materialize] = materialize_dense(source);
    dense_value.emplace(source, output);
    materialize->entries =
        source_entries & (source_entries - 1) ? uint64_t{1} : source_entries;
    plan.values[output].local_entries = materialize->entries;
    materializations.emplace(source, std::move(materialize));
  }
  for (const auto &[source, source_entries] : rhs_entries) {
    auto [output, materialize] = materialize_dense(source);
    dense_rhs.emplace(source, output);
    materialize->entries = source_entries;
    plan.values[output].local_entries = materialize->entries;
    rhs_materializations.emplace(source, std::move(materialize));
  }
  if (materializations.empty() && rhs_materializations.empty()) return;

  std::set<size_t> emitted, emitted_rhs;
  std::vector<std::unique_ptr<operation>> reordered;
  reordered.reserve(original_operations + materializations.size() +
                    rhs_materializations.size());
  for (size_t i = 0; i < original_operations; ++i) {
    auto &operation = plan.operations[i];
    if (auto *solve = dynamic_cast<solve_operation *>(operation.get())) {
      const size_t source = solve->matrix_value;
      if (const auto found = dense_value.find(source);
          found != dense_value.end()) {
        if (emitted.insert(source).second)
          reordered.push_back(std::move(materializations.at(source)));
        solve->matrix_value = found->second;
      }
      const size_t rhs = solve->rhs;
      if (const auto found = dense_rhs.find(rhs); found != dense_rhs.end()) {
        if (emitted_rhs.insert(rhs).second)
          reordered.push_back(std::move(rhs_materializations.at(rhs)));
        solve->rhs = found->second;
      }
    }
    reordered.push_back(std::move(operation));
  }
  plan.operations = std::move(reordered);
}

void build_entry_schedules(casadi_mx_graph_plan_impl &plan) {
  plan.panel_pointer_offsets.reserve(plan.values.size() + 1);
  plan.panel_pointer_offsets.push_back(0);
  for (const auto &value : plan.values)
    plan.panel_pointer_offsets.push_back(
        plan.panel_pointer_offsets.back() + value.matrix.panels.size());
  plan.value_panel_pointers = plan.panel_pointer_offsets.back();
  const size_t original_operations = plan.operations.size();
  const size_t pointer_count = plan.value_panel_pointers +
                               plan.external_inputs + plan.outputs;
  const auto slots = [&](size_t value) {
    std::vector<size_t> result(plan.values[value].matrix.panels.size());
    std::iota(result.begin(), result.end(),
              plan.panel_pointer_offsets[value]);
    return result;
  };
  for (size_t index = 0; index < original_operations; ++index) {
    auto *product =
        dynamic_cast<product_operation *>(plan.operations[index].get());
    if (!product) continue;
    const auto lhs_slots = slots(product->lhs);
    const auto rhs_slots = slots(product->rhs);
    const auto output_slots = slots(product->output);
    panel_program_spec initialization{.pointers = pointer_count};
    if (product->addend != product->output)
      for (size_t nz = 0; nz < product->addend_map.size(); ++nz)
        append_assignment(
            initialization, panel_operand(plan, product->output, nz),
            panel_operand(plan, product->addend, product->addend_map[nz]),
            product->addend_scale);
    product->kernel = compile_indexed_sparse_product_lazy(
        plan.values[product->lhs].matrix,
        plan.values[product->rhs].matrix, product_op::times,
        product->product_scale,
        plan.values[product->output].matrix, pointer_count,
        lhs_slots, rhs_slots, output_slots, std::move(initialization));
  }
  plan.entry_schedules.resize(plan.entry_outputs.size());
  for (size_t entry = 0; entry < plan.entry_outputs.size(); ++entry) {
    auto &schedule = plan.entry_schedules[entry];
    panel_program_spec program{.pointers = pointer_count};
    const auto flush = [&] {
      if (program.instructions.empty()) return;
      panel_program_spec retained = coalesce_panel_program_spec(program);
      auto compiled = std::make_unique<backend_program_operation>(
          std::move(retained), compile_panel_program(std::move(program)));
      compiled->entries = uint64_t{1} << entry;
      schedule.push_back(compiled.get());
      plan.operations.push_back(std::move(compiled));
      program = panel_program_spec{.pointers = pointer_count};
    };
    for (size_t index = 0; index < original_operations; ++index) {
      const auto &operation = *plan.operations[index];
      if (operation.binding_only() ||
          !(operation.entries & (uint64_t{1} << entry)))
        continue;
      if (lower_to_panel_program(plan, operation, program)) continue;
      flush();
      schedule.push_back(&operation);
    }
    flush();
  }
}

struct generated_source {
  std::ostringstream text;
  size_t indent = 0;

  void line(std::string_view value = {}) {
    for (size_t i = 0; i < indent; ++i) text << "  ";
    text << value << '\n';
  }
  void open(std::string_view value) {
    line(std::string(value) + " {");
    ++indent;
  }
  void close(std::string_view suffix = {}) {
    --indent;
    line(std::string("}") + std::string(suffix));
  }
};

struct generated_panel {
  panel_layout panel;
  bool transpose() const { return panel.transposed; }
  size_t row() const { return panel.row_offset; }
  size_t col() const { return panel.col_offset; }
  size_t rows() const { return panel.rows; }
  size_t cols() const { return panel.cols; }
  size_t physical_rows() const {
    return panel.transposed ? panel.cols : panel.rows;
  }
  size_t physical_cols() const {
    return panel.transposed ? panel.rows : panel.cols;
  }
  size_t leading_rows() const {
    return panel.storage_rows ? panel.storage_rows : physical_rows();
  }
  bool structured() const { return panel.pattern != sparsity::dense; }
};

char generated_alignment(const generated_panel &panel, size_t k,
                         bool lhs) {
  if (panel.panel.pattern == sparsity::eye) return 'x';
  size_t offset = k;
  if (panel.panel.pattern == sparsity::dense)
    offset = lhs ? (panel.transpose() ? k : k * panel.leading_rows())
                 : (panel.transpose() ? k * panel.leading_rows() : k);
  return (panel.panel.storage_offset + offset) * sizeof(scalar_t) %
                     EIGEN_MAX_ALIGN_BYTES ==
                 0
             ? 'a'
             : 'u';
}

std::string generated_panel_address(panel_program_operand value,
                                    std::string_view index = "0",
                                    ptrdiff_t outer_stride = 0,
                                    std::string_view outer_index = "0") {
  if (value.pointer == panel_program_operand::invalid) return "0.0";
  std::ostringstream result;
  result << "p[" << value.pointer << "][" << value.offset;
  if (value.stride) result << "+(" << value.stride << ")*" << index;
  if (outer_stride)
    result << "+(" << outer_stride << ")*" << outer_index;
  result << ']';
  return result.str();
}

std::string generated_number(scalar_t value) {
  std::ostringstream result;
  result << std::setprecision(std::numeric_limits<scalar_t>::max_digits10)
         << value;
  return result.str();
}

void emit_generated_panel_program(generated_source &source,
                                  const panel_program_spec &spec) {
  const auto same_shape = [](const panel_program_instruction &lhs,
                             const panel_program_instruction &rhs) {
    const auto same_operand = [](const panel_program_operand &a,
                                 const panel_program_operand &b) {
      return a.pointer == b.pointer && a.stride == b.stride;
    };
    return lhs.op == rhs.op && lhs.count == rhs.count &&
           lhs.scalar == rhs.scalar &&
           same_operand(lhs.destination, rhs.destination) &&
           same_operand(lhs.lhs, rhs.lhs) && same_operand(lhs.rhs, rhs.rhs);
  };
  const auto delta = [](const panel_program_operand &next,
                        const panel_program_operand &first) {
    return static_cast<ptrdiff_t>(next.offset) -
           static_cast<ptrdiff_t>(first.offset);
  };
  for (size_t begin = 0; begin < spec.instructions.size();) {
    const auto &instruction = spec.instructions[begin];
    size_t repeat = 1;
    ptrdiff_t destination_outer = 0, lhs_outer = 0, rhs_outer = 0;
    if (begin + 1 < spec.instructions.size() &&
        same_shape(instruction, spec.instructions[begin + 1])) {
      destination_outer = delta(spec.instructions[begin + 1].destination,
                                instruction.destination);
      lhs_outer = delta(spec.instructions[begin + 1].lhs, instruction.lhs);
      rhs_outer = delta(spec.instructions[begin + 1].rhs, instruction.rhs);
      for (size_t next = begin + 1; next < spec.instructions.size(); ++next) {
        const auto &candidate = spec.instructions[next];
        if (!same_shape(instruction, candidate) ||
            delta(candidate.destination, instruction.destination) !=
                destination_outer * static_cast<ptrdiff_t>(next - begin) ||
            delta(candidate.lhs, instruction.lhs) !=
                lhs_outer * static_cast<ptrdiff_t>(next - begin) ||
            delta(candidate.rhs, instruction.rhs) !=
                rhs_outer * static_cast<ptrdiff_t>(next - begin))
          break;
        repeat = next - begin + 1;
      }
    }
    if (repeat > 1)
      source.open("for (std::size_t j=0;j<" + std::to_string(repeat) +
                  ";++j)");
    const bool loop = instruction.count > 1;
    if (loop) {
      source.open("for (std::size_t i=0;i<" +
                  std::to_string(instruction.count) + ";++i)");
    }
    const std::string index = loop ? "i" : "0";
    const std::string outer_index = repeat > 1 ? "j" : "0";
    std::string expression;
    switch (instruction.op) {
    case panel_program_op::fill:
      expression = generated_number(instruction.scalar);
      break;
    case panel_program_op::copy:
      expression = generated_panel_address(instruction.lhs, index, lhs_outer,
                                           outer_index);
      if (instruction.scalar != 1.)
        expression = generated_number(instruction.scalar) + "*(" +
                     expression + ')';
      break;
    case panel_program_op::add:
    case panel_program_op::sub:
    case panel_program_op::mul:
    case panel_program_op::div:
      expression = generated_panel_address(instruction.lhs, index, lhs_outer,
                                           outer_index) +
                   (instruction.op == panel_program_op::add ? "+"
                    : instruction.op == panel_program_op::sub ? "-"
                    : instruction.op == panel_program_op::mul ? "*"
                                                               : "/") +
                   generated_panel_address(instruction.rhs, index, rhs_outer,
                                           outer_index);
      break;
    }
    source.line(generated_panel_address(instruction.destination, index,
                                        destination_outer, outer_index) +
                "=" + expression + ";");
    if (loop) source.close();
    if (repeat > 1) source.close();
    begin += repeat;
  }
}

void emit_generated_product(generated_source &source,
                            const casadi_mx_graph_plan_impl &plan,
                            const product_operation &operation) {
  const auto &lhs = plan.values[operation.lhs].matrix;
  const auto &rhs = plan.values[operation.rhs].matrix;
  const auto &output = plan.values[operation.output].matrix;
  panel_program_spec initialization{
      .pointers = plan.value_panel_pointers + plan.external_inputs +
                  plan.outputs};
  if (operation.addend != operation.output)
    for (size_t nz = 0; nz < operation.addend_map.size(); ++nz)
      append_assignment(initialization,
                        panel_operand(plan, operation.output, nz),
                        panel_operand(plan, operation.addend,
                                      operation.addend_map[nz]),
                        operation.addend_scale);
  emit_generated_panel_program(
      source, coalesce_panel_program_spec(std::move(initialization)));

  for (size_t li = 0; li < lhs.panels.size(); ++li) {
    const generated_panel l{lhs.panels[li]};
    for (size_t ri = 0; ri < rhs.panels.size(); ++ri) {
      const generated_panel r{rhs.panels[ri]};
      const size_t begin = std::max(l.col(), r.row());
      const size_t end = std::min(l.col() + l.cols(), r.row() + r.rows());
      if (end <= begin) continue;
      const size_t n = end - begin;
      const size_t lk = begin - l.col(), rk = begin - r.row();
      const size_t out_row = l.row() + (l.structured() ? lk : 0);
      const size_t out_col = r.col() + (r.structured() ? rk : 0);
      const size_t result_rows = l.structured() ? n : l.rows();
      const size_t result_cols = r.structured() ? n : r.cols();
      const bool diagonal_result = l.structured() && r.structured();
      size_t oi = 0, destination_rows = 0, destination_row = 0,
             destination_col = 0, destination_offset = 0;
      bool found = false;
      for (; oi < output.panels.size(); ++oi) {
        const auto &panel = output.panels[oi];
        if (panel.pattern == sparsity::dense &&
            panel.row_offset <= out_row && panel.col_offset <= out_col &&
            out_row + result_rows <= panel.row_offset + panel.rows &&
            out_col + result_cols <= panel.col_offset + panel.cols) {
          destination_rows = panel.rows;
          destination_row = out_row - panel.row_offset;
          destination_col = out_col - panel.col_offset;
          found = true;
          break;
        }
        if (panel.pattern == sparsity::diag && diagonal_result &&
            panel.row_offset <= out_row && panel.col_offset <= out_col &&
            out_row - panel.row_offset == out_col - panel.col_offset &&
            out_row + n <= panel.row_offset + panel.rows) {
          destination_offset = out_row - panel.row_offset;
          found = true;
          break;
        }
      }
      if (!found)
        throw std::runtime_error(
            "generated product contribution has no output panel");
      const size_t lhs_slot = plan.panel_pointer_offsets[operation.lhs] + li;
      const size_t rhs_slot = plan.panel_pointer_offsets[operation.rhs] + ri;
      const size_t output_slot =
          plan.panel_pointer_offsets[operation.output] + oi;
      const char la = generated_alignment(l, lk, true);
      const char ra = generated_alignment(r, rk, false);
      std::ostringstream call;
      unsigned flags = 0;
      if (!l.structured() && !r.structured()) {
        flags = unsigned(l.transpose()) | (unsigned(r.transpose()) << 1) |
                (unsigned(la == 'a') << 2) | (unsigned(ra == 'a') << 3);
        call << "moto_graph_pair_dd";
      } else if (l.structured() && !r.structured()) {
        flags = unsigned(l.panel.pattern == sparsity::eye) |
                (unsigned(r.transpose()) << 1) |
                (unsigned(ra == 'a') << 2);
        call << "moto_graph_pair_sd";
      } else if (!l.structured() && r.structured()) {
        flags = unsigned(l.transpose()) |
                (unsigned(r.panel.pattern == sparsity::eye) << 1) |
                (unsigned(la == 'a') << 2);
        call << "moto_graph_pair_ds";
      } else {
        flags = unsigned(l.panel.pattern == sparsity::eye) |
                (unsigned(r.panel.pattern == sparsity::eye) << 1) |
                (unsigned(la == 'a') << 2) | (unsigned(ra == 'a') << 3);
        call << "moto_graph_pair_ss";
      }
      call << '(' << flags << ",p[" << lhs_slot << "]+"
           << l.panel.storage_offset << ','
           << l.physical_rows() << ',' << l.physical_cols() << ','
           << l.leading_rows() << ',' << lk << ",p[" << rhs_slot << "]+"
           << r.panel.storage_offset << ',' << r.physical_rows() << ','
           << r.physical_cols() << ',' << r.leading_rows() << ',' << rk
           << ",p[" << output_slot << "]+" << destination_offset << ','
           << destination_rows << ',' << destination_row << ','
           << destination_col << ',' << n << ','
           << generated_number(operation.product_scale)
           << ");";
      source.line(call.str());
    }
  }
}

bool emit_generated_solve(generated_source &source,
                          const casadi_mx_graph_plan_impl &plan,
                          const solve_operation &operation) {
  const auto &factor_layout = plan.values[operation.matrix_value];
  if (!factor_layout.direct_csc_panel()) return false;
  const size_t n = factor_layout.matrix.rows;
  const size_t factor_pointer =
      plan.panel_pointer_offsets[operation.matrix_value];
  const auto &rhs = plan.values[operation.rhs];
  const auto &output = plan.values[operation.output];
  const sparse_index rhs_index(rhs.sparsity);
  const auto zero_output = [&] {
    for (size_t nz = 0; nz < output.csc.size(); ++nz) {
      const auto destination = panel_operand(plan, operation.output, nz);
      if (destination.pointer != panel_program_operand::invalid)
        source.line(generated_panel_address(destination) + "=0.0;");
    }
  };
  if (operation.active_columns.empty()) {
    zero_output();
    return true;
  }
  if (operation.identity_rhs && output.direct_csc_panel()) {
    const size_t output_pointer = plan.panel_pointer_offsets[operation.output];
    if (n <= 3) {
      source.line("moto_graph_small_inverse(p[" +
                  std::to_string(factor_pointer) + "],p[" +
                  std::to_string(output_pointer) + "]," +
                  std::to_string(n) + ");");
      return true;
    }
    source.line("moto_graph_factor(s," +
                std::to_string(operation.factor_slot) + "," +
                std::to_string(operation.spd) + ",p[" +
                std::to_string(factor_pointer) + "]," + std::to_string(n) +
                ");");
    source.line("moto_graph_inverse(s," +
                std::to_string(operation.factor_slot) + ",p[" +
                std::to_string(output_pointer) + "]," +
                std::to_string(n) + ");");
    return true;
  }
  if (rhs.direct_csc_panel() && output.direct_csc_panel() &&
      operation.active_columns.size() ==
          static_cast<size_t>(rhs_index.cols)) {
    const size_t rhs_pointer = plan.panel_pointer_offsets[operation.rhs];
    const size_t output_pointer = plan.panel_pointer_offsets[operation.output];
    if (n <= 3) {
      source.line("moto_graph_small_solve(p[" +
                  std::to_string(factor_pointer) + "]," +
                  std::to_string(operation.transpose) + ",p[" +
                  std::to_string(rhs_pointer) + "]," + std::to_string(n) +
                  "," + std::to_string(operation.active_columns.size()) +
                  "," + std::to_string(n) + ",p[" +
                  std::to_string(output_pointer) + "]," +
                  std::to_string(n) + ");");
      return true;
    }
    source.line("moto_graph_factor(s," +
                std::to_string(operation.factor_slot) + "," +
                std::to_string(operation.spd) + ",p[" +
                std::to_string(factor_pointer) + "]," + std::to_string(n) +
                ");");
    source.line("moto_graph_solve(s," +
                std::to_string(operation.factor_slot) + "," +
                std::to_string(operation.transpose) + ",p[" +
                std::to_string(rhs_pointer) + "]," + std::to_string(n) +
                "," + std::to_string(operation.active_columns.size()) +
                "," + std::to_string(n) + ",p[" +
                std::to_string(output_pointer) + "]," +
                std::to_string(n) + ");");
    return true;
  }
  if (operation.segments.empty()) {
    const size_t cols = operation.active_columns.size();
    const std::string tag = std::to_string(operation.output);
    const std::string packed_rhs = "solve_rhs_" + tag;
    const std::string packed_output = "solve_output_" + tag;
    source.line("alignas(64) double " + packed_rhs + "[" +
                std::to_string(n * cols) + "]{};");
    source.line("alignas(64) double " + packed_output + "[" +
                std::to_string(n * cols) + "];");
    for (size_t packed = 0; packed < cols; ++packed) {
      const index_t col = operation.active_columns[packed];
      for (index_t nz = rhs_index.colind[col];
           nz < rhs_index.colind[col + 1]; ++nz) {
        const auto source_operand = panel_operand(plan, operation.rhs, nz);
        if (source_operand.pointer == panel_program_operand::invalid) continue;
        source.line(packed_rhs + "[" +
                    std::to_string(rhs_index.row[nz] + packed * n) + "]=" +
                    generated_panel_address(source_operand) + ";");
      }
    }
    if (n <= 3) {
      source.line("moto_graph_small_solve(p[" +
                  std::to_string(factor_pointer) + "]," +
                  std::to_string(operation.transpose) + "," + packed_rhs +
                  "," + std::to_string(n) + "," + std::to_string(cols) +
                  "," + std::to_string(n) + "," + packed_output + "," +
                  std::to_string(n) + ");");
    } else {
      source.line("moto_graph_factor(s," +
                  std::to_string(operation.factor_slot) + "," +
                  std::to_string(operation.spd) + ",p[" +
                  std::to_string(factor_pointer) + "]," +
                  std::to_string(n) + ");");
      source.line("moto_graph_solve(s," +
                  std::to_string(operation.factor_slot) + "," +
                  std::to_string(operation.transpose) + "," + packed_rhs +
                  "," + std::to_string(n) + "," + std::to_string(cols) +
                  "," + std::to_string(n) + "," + packed_output + "," +
                  std::to_string(n) + ");");
    }
    std::vector<index_t> packed_column(rhs_index.cols, -1);
    for (size_t packed = 0; packed < cols; ++packed)
      packed_column[operation.active_columns[packed]] =
          static_cast<index_t>(packed);
    const sparse_index output_index(output.sparsity);
    for (index_t col = 0; col < output_index.cols; ++col) {
      for (index_t nz = output_index.colind[col];
           nz < output_index.colind[col + 1]; ++nz) {
        const auto destination = panel_operand(plan, operation.output, nz);
        if (destination.pointer == panel_program_operand::invalid) continue;
        const index_t packed = packed_column[col];
        if (packed < 0)
          source.line(generated_panel_address(destination) + "=0.0;");
        else
          source.line(generated_panel_address(destination) + "=" +
                      packed_output + "[" +
                      std::to_string(output_index.row[nz] + packed * n) +
                      "];");
      }
    }
    return true;
  }
  source.line("moto_graph_factor(s," +
              std::to_string(operation.factor_slot) + "," +
              std::to_string(operation.spd) + ",p[" +
              std::to_string(factor_pointer) + "]," + std::to_string(n) +
              ");");
  for (const auto &segment : operation.segments) {
    const size_t rhs_pointer = plan.panel_pointer_offsets[operation.rhs] +
                               segment.rhs_panel;
    const size_t output_pointer =
        plan.panel_pointer_offsets[operation.output] + segment.output_panel;
    source.line("moto_graph_solve(s," +
                std::to_string(operation.factor_slot) + "," +
                std::to_string(operation.transpose) + ",p[" +
                std::to_string(rhs_pointer) + "]+" +
                std::to_string(segment.rhs_offset) + "," +
                std::to_string(n) + "," + std::to_string(segment.cols) +
                "," + std::to_string(segment.rhs_leading) + ",p[" +
                std::to_string(output_pointer) + "]+" +
                std::to_string(segment.output_offset) + "," +
                std::to_string(segment.output_leading) + ");");
  }
  return true;
}

std::string generated_graph_preamble(size_t factors) {
  {
    std::ostringstream source;
    source << R"cpp(#include <cstddef>
extern "C" void moto_graph_pair_dd(unsigned,const double*,std::size_t,std::size_t,std::size_t,std::size_t,const double*,std::size_t,std::size_t,std::size_t,std::size_t,double*,std::size_t,std::size_t,std::size_t,std::size_t,double);
extern "C" void moto_graph_pair_sd(unsigned,const double*,std::size_t,std::size_t,std::size_t,std::size_t,const double*,std::size_t,std::size_t,std::size_t,std::size_t,double*,std::size_t,std::size_t,std::size_t,std::size_t,double);
extern "C" void moto_graph_pair_ds(unsigned,const double*,std::size_t,std::size_t,std::size_t,std::size_t,const double*,std::size_t,std::size_t,std::size_t,std::size_t,double*,std::size_t,std::size_t,std::size_t,std::size_t,double);
extern "C" void moto_graph_pair_ss(unsigned,const double*,std::size_t,std::size_t,std::size_t,std::size_t,const double*,std::size_t,std::size_t,std::size_t,std::size_t,double*,std::size_t,std::size_t,std::size_t,std::size_t,double);
extern "C" void* moto_graph_factor_state_create(std::size_t);
extern "C" void moto_graph_factor_state_destroy(void*);
extern "C" void moto_graph_factor_next_epoch(void*);
extern "C" void moto_graph_small_inverse(const double*,double*,std::size_t);
extern "C" void moto_graph_small_solve(const double*,bool,const double*,std::size_t,std::size_t,std::size_t,double*,std::size_t);
extern "C" void moto_graph_factor(void*,std::size_t,bool,const double*,std::size_t);
extern "C" void moto_graph_inverse(void*,std::size_t,double*,std::size_t);
extern "C" void moto_graph_solve(void*,std::size_t,bool,const double*,std::size_t,std::size_t,std::size_t,double*,std::size_t);
)cpp";
    return source.str();
  }
}

casadi_mx_graph_plan_impl::whole_kernel_registry compile_generated_graph(
    const casadi_mx_graph_plan_impl &plan,
    const std::filesystem::path &cache_dir) {
  generated_source source;
  source.text << generated_graph_preamble(plan.factor_values.size());
  source.open("static void run(void*s,std::size_t entry,double**p)");
  source.line("if(entry==0)moto_graph_factor_next_epoch(s);");
  source.open("switch(entry)");
  for (size_t entry = 0; entry < plan.entry_schedules.size(); ++entry) {
    source.open("case " + std::to_string(entry) + ":");
    for (size_t value = 0; value < plan.values.size(); ++value) {
      const auto &layout = plan.values[value];
      if (!(layout.local_entries & (uint64_t{1} << entry))) continue;
      if (!layout.direct_csc_panel())
        throw std::logic_error(
            "generated local graph temporary must be one dense panel");
      const std::string name = "tmp" + std::to_string(value);
      source.line("alignas(64) double " + name + '[' +
                  std::to_string(layout.csc.size()) + "];");
      source.line("p[" + std::to_string(plan.panel_pointer_offsets[value]) +
                  "]=" + name + ";");
    }
    for (const operation *base : plan.entry_schedules[entry]) {
      if (const auto *program =
              dynamic_cast<const backend_program_operation *>(base)) {
        emit_generated_panel_program(source, program->spec);
      } else if (const auto *product =
                     dynamic_cast<const product_operation *>(base)) {
        emit_generated_product(source, plan, *product);
      } else if (const auto *solve =
                     dynamic_cast<const solve_operation *>(base)) {
        if (!emit_generated_solve(source, plan, *solve)) {
          if (std::getenv("MOTO_TRACE_GRAPH_LOWERING"))
            std::cerr << "whole graph JIT rejected solve: matrix_direct="
                      << plan.values[solve->matrix_value].direct_csc_panel()
                      << " rhs_direct="
                      << plan.values[solve->rhs].direct_csc_panel()
                      << " output_direct="
                      << plan.values[solve->output].direct_csc_panel()
                      << " segments=" << solve->segments.size()
                      << " active_columns=" << solve->active_columns.size()
                      << '\n';
          return nullptr;
        }
      } else {
        if (std::getenv("MOTO_TRACE_GRAPH_LOWERING"))
          std::cerr << "whole graph JIT rejected operation\n";
        return nullptr;
      }
    }
    source.line("break;");
    source.close();
  }
  source.close();
  source.close();
  source.open("extern \"C\" __attribute__((visibility(\"default\"))) void* moto_linear_jit_kernel(std::size_t action,void*opaque,std::size_t entry,double**p)");
  source.line("if(action==0)return moto_graph_factor_state_create(" +
              std::to_string(plan.factor_values.size()) + ");");
  source.line("if(action==1){moto_graph_factor_state_destroy(opaque);return nullptr;}");
  source.line("run(opaque,entry,p);return nullptr;");
  source.close();
  return reinterpret_cast<casadi_mx_graph_plan_impl::whole_kernel_registry>(
      compile_casadi_mx_graph_source(source.text.str(), cache_dir));
}

std::vector<index_t> transpose_map(const casadi::Sparsity &output,
                                   const casadi::Sparsity &input) {
  const sparse_index out(output), in(input);
  std::vector<index_t> map;
  map.reserve(out.row.size());
  for (index_t col = 0; col < out.cols; ++col)
    for (index_t nz = out.colind[col]; nz < out.colind[col + 1]; ++nz)
      map.push_back(in.find(col, out.row[nz]));
  return map;
}

void require_map(const std::vector<index_t> &map, std::string_view what) {
  if (std::ranges::find(map, -1) != map.end())
    throw std::runtime_error(std::string(what) + " sparsity mismatch");
}

} // namespace

struct casadi_mx_graph_plan : casadi_mx_graph_plan_impl {};

std::shared_ptr<const casadi_mx_graph_plan>
translate_casadi_mx_graph(const casadi::Function &function,
                          std::vector<size_t> entry_outputs,
                          std::span<const matrix_layout> input_layouts,
                          const std::filesystem::path &cache_dir,
                          size_t spd_outputs) {
  auto plan = std::make_shared<casadi_mx_graph_plan>();
  plan->inputs = function.n_in();
  if (spd_outputs > static_cast<size_t>(function.n_out()))
    throw std::invalid_argument("invalid SPD metadata output count");
  plan->outputs = function.n_out() - spd_outputs;
  if (!input_layouts.empty() && input_layouts.size() != plan->inputs)
    throw std::invalid_argument(
        "MX graph input-layout count does not match logical inputs");
  std::vector<value_layout> runtime_inputs;
  std::vector<bool> direct_inputs;
  runtime_inputs.reserve(plan->inputs);
  direct_inputs.reserve(plan->inputs);
  plan->input_pointer_offsets.push_back(0);
  for (size_t input = 0; input < plan->inputs; ++input) {
    value_layout layout = make_layout(function.sparsity_in(input));
    const bool exact_layout = !input_layouts.empty() &&
                              (input_layouts[input].rows ||
                               input_layouts[input].cols);
    if (exact_layout) {
      if (input_layouts[input].rows !=
              static_cast<size_t>(function.size1_in(input)) ||
          input_layouts[input].cols !=
              static_cast<size_t>(function.size2_in(input)))
        throw std::invalid_argument("MX graph input layout shape mismatch");
      layout.matrix = input_layouts[input];
      rebuild_csc_map(layout);
    }
    const bool direct = exact_layout || layout.direct_csc_panel();
    plan->external_inputs +=
        exact_layout ? layout.matrix.panels.size() : size_t{1};
    plan->input_pointer_offsets.push_back(plan->external_inputs);
    runtime_inputs.push_back(std::move(layout));
    direct_inputs.push_back(direct);
  }
  plan->input_values.assign(plan->inputs, no_value);
  if (entry_outputs.empty()) entry_outputs.push_back(plan->outputs);
  if (entry_outputs.size() > 64 ||
      std::accumulate(entry_outputs.begin(), entry_outputs.end(), size_t{0}) !=
          plan->outputs)
    throw std::invalid_argument("invalid CasADi MX graph entry layout");
  plan->entry_outputs = std::move(entry_outputs);
  plan->output_entries.resize(function.n_out());
  size_t output_cursor = 0;
  for (size_t entry = 0; entry < plan->entry_outputs.size(); ++entry)
    for (size_t i = 0; i < plan->entry_outputs[entry]; ++i)
      plan->output_entries[output_cursor++] = entry;
  plan->reference = function;
  std::vector<size_t> current;
  std::vector<size_t> output_offset(function.n_out());
  index_t maximum_slot = -1;
  for (index_t instruction = 0; instruction < function.n_instructions();
       ++instruction) {
    for (index_t slot : function.instruction_input(instruction))
      maximum_slot = std::max(maximum_slot, slot);
    if (function.instruction_id(instruction) != casadi::OP_OUTPUT)
      for (index_t slot : function.instruction_output(instruction))
        maximum_slot = std::max(maximum_slot, slot);
  }
  current.assign(maximum_slot + 1, no_value);

  std::vector<std::vector<index_t>> instruction_dependencies(
      function.n_instructions());
  std::vector<uint64_t> instruction_entries(function.n_instructions(), 0);
  std::vector<index_t> slot_producer(maximum_slot + 1, -1);
  for (index_t instruction = 0; instruction < function.n_instructions();
       ++instruction) {
    for (const index_t slot : function.instruction_input(instruction))
      if (slot >= 0 && slot_producer.at(slot) >= 0)
        instruction_dependencies[instruction].push_back(
            slot_producer[slot]);
    if (function.instruction_id(instruction) == casadi::OP_OUTPUT) {
      const auto output = function.instruction_output(instruction);
      if (output.size() == 1 && output[0] >= 0)
        instruction_entries[instruction] |=
            uint64_t{1} << plan->output_entries.at(output[0]);
    } else {
      for (const index_t slot : function.instruction_output(instruction))
        if (slot >= 0) slot_producer.at(slot) = instruction;
    }
  }
  for (index_t instruction = function.n_instructions(); instruction-- > 0;)
    for (const index_t dependency : instruction_dependencies[instruction])
      instruction_entries[dependency] |= instruction_entries[instruction];

  const auto add_value = [&](const casadi::Sparsity &sp) {
    plan->values.push_back(make_layout(sp));
    return plan->values.size() - 1;
  };
  const auto fail = [&](index_t instruction, std::string message) -> void {
    throw std::runtime_error("sparse MX graph instruction " +
                             std::to_string(instruction) + ": " + message);
  };
  for (index_t instruction = 0; instruction < function.n_instructions();
       ++instruction) {
    operation_entry_scope entry_scope{
        *plan, plan->operations.size(), instruction_entries[instruction]};
    const index_t op = function.instruction_id(instruction);
    const auto input_slots = function.instruction_input(instruction);
    const auto output_slots = function.instruction_output(instruction);
    const casadi::MX node = function.instruction_MX(instruction);
    std::vector<size_t> inputs;
    inputs.reserve(input_slots.size());
    for (index_t slot : input_slots) {
      if (slot < 0) inputs.push_back(no_value);
      else if (op == casadi::OP_INPUT) inputs.push_back(no_value);
      else if (current.at(slot) == no_value) fail(instruction, "unbound input");
      else inputs.push_back(current[slot]);
    }

    if (op == casadi::OP_OUTPUT) {
      if (inputs.size() != 1 || output_slots.size() != 1 ||
          inputs[0] == no_value || output_slots[0] < 0)
        fail(instruction, "invalid output");
      const size_t function_output = node.info().at("ind").to_int();
      if (function_output >= plan->outputs) {
        plan->spd_values.push_back(inputs[0]);
        continue;
      }
      const size_t output = output_slots[0];
      const size_t count = plan->values[inputs[0]].csc.size();
      plan->operations.push_back(std::make_unique<output_operation>(
          output_operation{inputs[0], output, output_offset[output]}));
      output_offset[output] += count;
      continue;
    }

    std::vector<size_t> outputs(output_slots.size(), no_value);
    for (size_t i = 0; i < output_slots.size(); ++i) {
      if (output_slots[i] < 0) continue;
      const auto sp = output_slots.size() == 1
                          ? node.sparsity()
                          : node.get_output(i).sparsity();
      outputs[i] = add_value(sp);
    }
    for (size_t i = 0; i < outputs.size(); ++i) {
      if (outputs[i] == no_value) continue;
      std::ostringstream fingerprint;
      fingerprint << op << ':' << i << ':' << node << ':';
      for (const size_t input : inputs)
        if (input != no_value)
          fingerprint << plan->values[input].fingerprint << ';';
      const std::string digest =
          utils::compute_md5_from_bytes(fingerprint.str());
      plan->values[outputs[i]].fingerprint = digest;
    }
    for (size_t i = 0; i < output_slots.size(); ++i)
      if (output_slots[i] >= 0) current[output_slots[i]] = outputs[i];
    if (op == casadi::OP_INPUT) {
      const auto info = node.info();
      const size_t input = info.at("ind").to_int();
      const size_t offset = info.at("offset").to_int();
      if (offset != 0 || node.nnz() != function.nnz_in(input))
        fail(instruction, "split MX inputs are not supported");
      plan->values[outputs[0]] = runtime_inputs.at(input);
      plan->values[outputs[0]].allocate = !direct_inputs[input];
      plan->input_values[input] = outputs[0];
      plan->operations.push_back(std::make_unique<input_operation>(
          input_operation{outputs[0], input, direct_inputs[input]}));
      continue;
    }
    if (op == casadi::OP_CONST) {
      plan->operations.push_back(std::make_unique<constant_operation>(
          constant_operation{outputs[0],
                             casadi::MX::evalf(node).nonzeros()}));
      continue;
    }
    if (op == casadi::OP_TRANSPOSE) {
      const auto &input_layout = plan->values[inputs[0]];
      auto &output_layout = plan->values[outputs[0]];
      output_layout.matrix.rows = input_layout.matrix.cols;
      output_layout.matrix.cols = input_layout.matrix.rows;
      output_layout.matrix.panels.clear();
      output_layout.matrix.panels.reserve(input_layout.matrix.panels.size());
      std::vector<panel_alias> aliases;
      aliases.reserve(input_layout.matrix.panels.size());
      for (size_t panel = 0; panel < input_layout.matrix.panels.size();
           ++panel) {
        const auto &source = input_layout.matrix.panels[panel];
        output_layout.matrix.panels.push_back(
            {source.pattern, source.col_offset, source.row_offset,
             source.cols, source.rows, !source.transposed,
             source.storage_offset, source.storage_rows});
        aliases.push_back({panel, inputs[0], panel, 0});
      }
      output_layout.virtual_zeros = input_layout.virtual_zeros;
      rebuild_csc_map(output_layout);
      output_layout.allocate = false;
      plan->operations.push_back(std::make_unique<alias_operation>(
          outputs[0], std::move(aliases)));
      continue;
    }
    if (op == casadi::OP_PROJECT &&
        contains_sparsity(node.sparsity(), node.dep(0).sparsity())) {
      const auto &input_layout = plan->values[inputs[0]];
      auto &output_layout = plan->values[outputs[0]];
      output_layout.matrix = input_layout.matrix;
      output_layout.virtual_zeros =
          input_layout.virtual_zeros ||
          node.sparsity().nnz() != node.dep(0).sparsity().nnz();
      rebuild_csc_map(output_layout);
      output_layout.allocate = false;
      std::vector<panel_alias> aliases;
      aliases.reserve(input_layout.matrix.panels.size());
      for (size_t panel = 0; panel < input_layout.matrix.panels.size(); ++panel)
        aliases.push_back({panel, inputs[0], panel, 0});
      plan->operations.push_back(std::make_unique<alias_operation>(
          outputs[0], std::move(aliases)));
      continue;
    }
    if (op == casadi::OP_HORZREPMAT) {
      const auto &input_layout = plan->values[inputs[0]];
      auto &output_layout = plan->values[outputs[0]];
      if (!input_layout.matrix.cols ||
          output_layout.matrix.cols % input_layout.matrix.cols)
        fail(instruction, "invalid horizontal repetition shape");
      const size_t copies =
          output_layout.matrix.cols / input_layout.matrix.cols;
      output_layout.matrix.panels.clear();
      std::vector<panel_alias> aliases;
      output_layout.matrix.panels.reserve(
          copies * input_layout.matrix.panels.size());
      aliases.reserve(copies * input_layout.matrix.panels.size());
      for (size_t copy = 0; copy < copies; ++copy)
        for (size_t panel = 0; panel < input_layout.matrix.panels.size();
             ++panel) {
          auto repeated = input_layout.matrix.panels[panel];
          repeated.col_offset += copy * input_layout.matrix.cols;
          output_layout.matrix.panels.push_back(repeated);
          aliases.push_back({output_layout.matrix.panels.size() - 1,
                             inputs[0], panel, 0});
        }
      output_layout.virtual_zeros = input_layout.virtual_zeros;
      rebuild_csc_map(output_layout);
      output_layout.allocate = false;
      plan->operations.push_back(std::make_unique<alias_operation>(
          outputs[0], std::move(aliases)));
      continue;
    }
    if (op == casadi::OP_ASSIGN || op == casadi::OP_NEG ||
        op == casadi::OP_PROJECT || op == casadi::OP_RESHAPE ||
        op == casadi::OP_SPARSITY_CAST || op == casadi::OP_HORZREPMAT ||
        op == casadi::OP_TRANSPOSE || op == casadi::OP_GETNONZEROS) {
      std::vector<index_t> map;
      if (op == casadi::OP_GETNONZEROS)
        map = nonzero_indices(node.info(), node.nnz());
      else if (op == casadi::OP_RESHAPE ||
               op == casadi::OP_SPARSITY_CAST) {
        map.resize(node.nnz());
        std::iota(map.begin(), map.end(), 0);
      }
      else if (op == casadi::OP_HORZREPMAT) {
        const sparse_index out(node.sparsity()), in(node.dep(0).sparsity());
        for (index_t col = 0; col < out.cols; ++col)
          for (index_t nz = out.colind[col]; nz < out.colind[col + 1]; ++nz)
            map.push_back(in.find(out.row[nz], col % in.cols));
      } else
        map = coordinate_map(node.sparsity(), node.dep(0).sparsity());
      if (op != casadi::OP_ASSIGN && op != casadi::OP_PROJECT &&
          op != casadi::OP_RESHAPE && op != casadi::OP_SPARSITY_CAST)
        require_map(map, "unary graph operation");
      append_alias_or_copy(*plan, inputs[0], outputs[0], std::move(map),
                           op == casadi::OP_NEG ? -1. : 1., op);
      continue;
    }
    if (op == casadi::OP_ADD || op == casadi::OP_SUB ||
        op == casadi::OP_MUL || op == casadi::OP_DIV) {
      auto lhs = coordinate_map(node.sparsity(), node.dep(0).sparsity());
      auto rhs = coordinate_map(node.sparsity(), node.dep(1).sparsity());
      if (op == casadi::OP_DIV) require_map(rhs, "division");
      plan->operations.push_back(std::make_unique<binary_operation>(
          binary_operation{inputs[0], inputs[1], outputs[0], op,
                           std::move(lhs), std::move(rhs)}));
      continue;
    }
    if (op == casadi::OP_MTIMES) {
      if (inputs.size() != 3) fail(instruction, "invalid MTIMES");
      ensure_product_coverage(plan->values[outputs[0]],
                              plan->values[inputs[1]],
                              plan->values[inputs[2]]);
      plan->operations.push_back(std::make_unique<product_operation>(
          product_operation{
              inputs[0], inputs[1], inputs[2], outputs[0], {},
              coordinate_map(node.sparsity(), node.dep(0).sparsity())}));
      continue;
    }
    if (op == casadi::OP_INVERSE)
      fail(instruction,
           "explicit inverse is not supported; use solve(A, B)");
    if (op == casadi::OP_SOLVE) {
      bool transpose = false;
      const auto info = node.info();
      if (const auto found = info.find("tr"); found != info.end())
        transpose = found->second.to_bool();
      bool identity_rhs = false;
      const auto rhs = node.dep(0);
      if (rhs.is_constant() && rhs.size1() == rhs.size2()) {
        const casadi::DM value = casadi::MX::evalf(rhs);
        identity_rhs = true;
        for (casadi_int col = 0; col < rhs.size2() && identity_rhs; ++col)
          for (casadi_int row = 0; row < rhs.size1(); ++row)
            if (value(row, col).scalar() != (row == col ? 1. : 0.)) {
              identity_rhs = false;
              break;
            }
      }
      plan->operations.push_back(std::make_unique<solve_operation>(
          solve_operation{inputs[0], inputs[1], outputs[0], transpose,
                          identity_rhs}));
      continue;
    }
    if (op == casadi::OP_HORZCAT || op == casadi::OP_VERTCAT ||
        op == casadi::OP_DIAGCAT) {
      const sparse_index out(node.sparsity());
      std::vector<std::vector<index_t>> destinations;
      size_t row_offset = 0, col_offset = 0;
      for (size_t dep = 0; dep < inputs.size(); ++dep) {
        const sparse_index in(node.dep(dep).sparsity());
        const size_t row_shift = op == casadi::OP_HORZCAT ? 0 : row_offset;
        const size_t col_shift = op == casadi::OP_VERTCAT ? 0 : col_offset;
        auto &map = destinations.emplace_back();
        for (index_t col = 0; col < in.cols; ++col)
          for (index_t nz = in.colind[col]; nz < in.colind[col + 1]; ++nz)
            map.push_back(out.find(in.row[nz] + row_shift, col + col_shift));
        require_map(map, "concatenation");
        if (op != casadi::OP_HORZCAT) row_offset += in.rows;
        if (op != casadi::OP_VERTCAT) col_offset += in.cols;
      }
      auto &output_layout = plan->values[outputs[0]];
      output_layout.matrix.panels.clear();
      row_offset = 0;
      col_offset = 0;
      for (const size_t input : inputs) {
        const auto &input_layout = plan->values[input].matrix;
        const size_t row_shift = op == casadi::OP_HORZCAT ? 0 : row_offset;
        const size_t col_shift = op == casadi::OP_VERTCAT ? 0 : col_offset;
        for (const auto &panel : input_layout.panels)
          output_layout.matrix.panels.push_back({
              panel.pattern, panel.row_offset + row_shift,
              panel.col_offset + col_shift, panel.rows, panel.cols,
              panel.transposed, panel.storage_offset, panel.storage_rows});
        if (op != casadi::OP_HORZCAT) row_offset += input_layout.rows;
        if (op != casadi::OP_VERTCAT) col_offset += input_layout.cols;
      }
      output_layout.virtual_zeros = std::ranges::any_of(
          inputs, [&](size_t input) {
            return plan->values[input].virtual_zeros;
          });
      rebuild_csc_map(output_layout);
      std::vector<source_nz> source(plan->values[outputs[0]].csc.size());
      for (size_t dep = 0; dep < inputs.size(); ++dep)
        for (size_t nz = 0; nz < destinations[dep].size(); ++nz) {
          const index_t destination = destinations[dep][nz];
          if (destination >= 0)
            source[destination] = {inputs[dep], static_cast<index_t>(nz)};
        }
      if (auto aliases = make_panel_aliases(*plan, outputs[0], source)) {
        plan->values[outputs[0]].allocate = false;
        plan->operations.push_back(std::make_unique<alias_operation>(
            outputs[0], std::move(*aliases)));
      } else {
        const std::string fingerprint = output_layout.fingerprint;
        output_layout = make_layout(node.sparsity());
        output_layout.fingerprint = fingerprint;
        plan->operations.push_back(std::make_unique<concat_operation>(
            concat_operation{outputs[0], inputs,
                             std::move(destinations)}));
      }
      continue;
    }
    if (op == casadi::OP_HORZSPLIT || op == casadi::OP_VERTSPLIT ||
        op == casadi::OP_DIAGSPLIT) {
      const sparse_index in(node.dep(0).sparsity());
      size_t row_offset = 0, col_offset = 0;
      for (size_t ordinal = 0; ordinal < outputs.size(); ++ordinal) {
        const auto value = node.get_output(ordinal);
        const sparse_index out(value.sparsity());
        if (outputs[ordinal] != no_value) {
          std::vector<index_t> map;
          const size_t row_shift =
              op == casadi::OP_HORZSPLIT ? 0 : row_offset;
          const size_t col_shift =
              op == casadi::OP_VERTSPLIT ? 0 : col_offset;
          for (index_t col = 0; col < out.cols; ++col)
            for (index_t nz = out.colind[col]; nz < out.colind[col + 1]; ++nz)
              map.push_back(in.find(out.row[nz] + row_shift,
                                    col + col_shift));
          require_map(map, "split");
          append_alias_or_copy(*plan, inputs[0], outputs[ordinal],
                               std::move(map), 1., op);
        }
        if (op != casadi::OP_HORZSPLIT) row_offset += out.rows;
        if (op != casadi::OP_VERTSPLIT) col_offset += out.cols;
      }
      continue;
    }
    if (op == casadi::OP_ADDNONZEROS || op == casadi::OP_SETNONZEROS) {
      bool add = op == casadi::OP_ADDNONZEROS;
      const auto info = node.info();
      if (const auto found = info.find("add"); found != info.end())
        add = found->second.to_bool();
      plan->operations.push_back(std::make_unique<scatter_operation>(
          scatter_operation{
              inputs[0], inputs[1], outputs[0],
              coordinate_map(node.sparsity(), node.dep(0).sparsity()),
              nonzero_indices(info, node.dep(1).nnz()), add,
              node.dep(1).nnz() == 1}));
      continue;
    }
    fail(instruction, "unsupported MX operation " + std::to_string(op));
  }
  std::set<std::string> spd_fingerprints, spd_matrix_fingerprints;
  for (const size_t value : plan->spd_values)
    spd_fingerprints.insert(plan->values[value].fingerprint);
  for (const auto &operation : plan->operations)
    if (const auto *solve =
            dynamic_cast<const solve_operation *>(operation.get());
        solve &&
        spd_fingerprints.contains(plan->values[solve->output].fingerprint))
      spd_matrix_fingerprints.insert(
          plan->values[solve->matrix_value].fingerprint);
  lazily_apply_inverses(*plan);
  // An SSA value required by several runtime entries is a persistent branch
  // point.  Materialize every such branch during presolve: action and
  // transpose-action are independently callable, so neither may depend on
  // the other having run first.
  for (const auto &operation : plan->operations) {
    if (operation->binding_only() || !operation->entries) continue;
    if (operation->entries & (operation->entries - 1))
      operation->entries = uint64_t{1};
  }
  fuse_lazy_products(*plan);
  fuse_product_accumulations(*plan);
  reuse_dense_output_materializations(*plan);
  materialize_reused_product_branches(*plan);
  // Preserve composite MX views throughout algebraic lowering.  A dense
  // factorization and its batched dense RHS are the boundaries that require
  // contiguous storage, so materialize them exactly once immediately before
  // their first solve instead of densifying every upstream block operation.
  materialize_solve_inputs(*plan);
  if (std::getenv("MOTO_TRACE_GRAPH_LOWERING")) {
    for (const auto &operation : plan->operations) {
      const auto *product =
          dynamic_cast<const product_operation *>(operation.get());
      if (!product || !(product->entries & uint64_t{1})) continue;
      const auto &lhs = plan->values[product->lhs].matrix;
      const auto &rhs = plan->values[product->rhs].matrix;
      size_t pairs = 0;
      for (const auto &l : lhs.panels)
        for (const auto &r : rhs.panels)
          pairs += std::max(l.col_offset, r.row_offset) <
                   std::min(l.col_offset + l.cols,
                            r.row_offset + r.rows);
      std::cerr << "presolve product " << product->lhs << '['
                << lhs.rows << 'x' << lhs.cols << ",p="
                << lhs.panels.size() << "] * " << product->rhs << '['
                << rhs.rows << 'x' << rhs.cols << ",p="
                << rhs.panels.size() << "] -> " << product->output
                << " pairs=" << pairs << '\n';
    }
  }
  if (std::getenv("MOTO_TRACE_GRAPH_FACTORS")) {
    for (const size_t value : plan->spd_values)
      std::cerr << "graph SPD value " << value << " shape "
                << plan->values[value].matrix.rows << 'x'
                << plan->values[value].matrix.cols << " fp "
                << plan->values[value].fingerprint << '\n';
    for (const auto &operation : plan->operations)
      if (const auto *solve =
              dynamic_cast<const solve_operation *>(operation.get()))
        std::cerr << "graph solve matrix " << solve->matrix_value
                  << " output " << solve->output << " shape "
                  << plan->values[solve->output].matrix.rows << 'x'
                  << plan->values[solve->output].matrix.cols << " matrix-fp "
                  << plan->values[solve->matrix_value].fingerprint
                  << " output-fp "
                  << plan->values[solve->output].fingerprint << '\n';
  }
  std::unordered_map<std::string, size_t> factor_slot_by_value;
  size_t solve_workspace = 0;
  for (const auto &operation : plan->operations) {
    auto *solve = dynamic_cast<solve_operation *>(operation.get());
    if (!solve) continue;
    const auto [found, inserted] = factor_slot_by_value.emplace(
        plan->values[solve->matrix_value].fingerprint,
        plan->factor_values.size());
    solve->spd = spd_matrix_fingerprints.contains(
        plan->values[solve->matrix_value].fingerprint);
    if (inserted) {
      plan->factor_values.push_back(solve->matrix_value);
      plan->factor_spd.push_back(solve->spd);
    } else if (plan->factor_spd[found->second] != solve->spd) {
      throw std::logic_error("inconsistent factorization property");
    }
    solve->factor_slot = found->second;
  }
  for (const auto &operation : plan->operations) {
    auto *solve = dynamic_cast<solve_operation *>(operation.get());
    if (!solve) continue;
    solve->workspace_slot = solve_workspace++;
    const sparse_index rhs(plan->values[solve->rhs].sparsity);
    for (index_t col = 0; col < rhs.cols; ++col)
      if (rhs.colind[col] != rhs.colind[col + 1])
        solve->active_columns.push_back(col);
    if (solve->active_columns.size() == static_cast<size_t>(rhs.cols)) {
      const auto &rhs_layout = plan->values[solve->rhs].matrix;
      const auto &output_layout = plan->values[solve->output].matrix;
      size_t covered_columns = 0;
      for (size_t ri = 0; ri < rhs_layout.panels.size(); ++ri) {
        const auto &source = rhs_layout.panels[ri];
        if (source.pattern != sparsity::dense || source.transposed ||
            source.row_offset != 0 ||
            source.rows != static_cast<size_t>(rhs.rows)) {
          solve->segments.clear();
          covered_columns = 0;
          break;
        }
        const auto found = std::ranges::find_if(
            output_layout.panels, [&](const panel_layout &destination) {
              return destination.pattern == sparsity::dense &&
                     !destination.transposed && destination.row_offset == 0 &&
                     destination.rows == static_cast<size_t>(rhs.rows) &&
                     destination.col_offset <= source.col_offset &&
                     source.col_offset + source.cols <=
                         destination.col_offset + destination.cols;
            });
        if (found == output_layout.panels.end()) {
          solve->segments.clear();
          covered_columns = 0;
          break;
        }
        const size_t oi = found - output_layout.panels.begin();
        const size_t source_leading =
            source.storage_rows ? source.storage_rows : source.rows;
        const size_t output_leading =
            found->storage_rows ? found->storage_rows : found->rows;
        solve->segments.push_back(
            {.rhs_panel = ri,
             .output_panel = oi,
             .rhs_offset = source.storage_offset,
             .output_offset = found->storage_offset +
                              (source.col_offset - found->col_offset) *
                                  output_leading,
             .rhs_leading = source_leading,
             .output_leading = output_leading,
             .cols = source.cols});
        covered_columns += source.cols;
      }
      if (covered_columns != static_cast<size_t>(rhs.cols))
        solve->segments.clear();
    }
  }
  for (size_t output = 0; output < plan->outputs; ++output)
    if (output_offset[output] != static_cast<size_t>(function.nnz_out(output)))
      throw std::runtime_error("incomplete sparse graph output");
  // Route complete result panels directly into caller-owned output storage.
  // MX frequently leaves a chain of slice/reshape aliases in front of an
  // output; copying that view nonzero-by-nonzero is unnecessary when it still
  // covers one complete producer panel.
  std::unordered_map<size_t, const alias_operation *> aliases_by_output;
  for (const auto &operation : plan->operations)
    if (const auto *alias =
            dynamic_cast<const alias_operation *>(operation.get()))
      aliases_by_output.emplace(alias->output, alias);
  std::set<std::pair<size_t, size_t>> bound_output_panels;
  for (const auto &operation : plan->operations) {
    auto *output = dynamic_cast<output_operation *>(operation.get());
    if (!output || !plan->values[output->input].direct_csc_panel()) continue;
    size_t value = output->input, panel = 0;
    const size_t elements = plan->values[value].csc.size();
    while (!plan->values[value].allocate) {
      const auto found = aliases_by_output.find(value);
      if (found == aliases_by_output.end()) break;
      const auto &aliases = found->second->aliases;
      const auto alias = std::ranges::find_if(
          aliases, [&](const panel_alias &entry) {
            return entry.output_panel == panel;
          });
      if (alias == aliases.end() || alias->input_offset != 0) break;
      const auto &source_panel =
          plan->values[alias->input_value].matrix.panels[alias->input_panel];
      const size_t source_elements =
          source_panel.pattern == sparsity::dense
              ? source_panel.rows * source_panel.cols
              : source_panel.rows;
      if (source_elements != elements) break;
      value = alias->input_value;
      panel = alias->input_panel;
    }
    if (!plan->values[value].allocate ||
        !bound_output_panels.emplace(value, panel).second)
      continue;
    plan->direct_outputs.push_back(
        {value, panel, output->output, output->offset});
    output->direct = true;
  }
  build_entry_schedules(*plan);
  if (!std::getenv("MOTO_DISABLE_LINEAR_GRAPH_WHOLE_JIT"))
    plan->whole_kernel = compile_generated_graph(*plan, cache_dir);
  return plan;
}

casadi_mx_graph_instance::casadi_mx_graph_instance(
    std::shared_ptr<const casadi_mx_graph_plan> input_plan,
    std::vector<sparse_matrix> *external_workspace)
    : plan(std::move(input_plan)) {
  if (!external_workspace) {
    owned_workspace = std::make_shared<std::vector<sparse_matrix>>();
    external_workspace = owned_workspace.get();
  }
  workspace = external_workspace;
  workspace_slots.reserve(plan->values.size());
  for (const auto &layout : plan->values) {
    if (layout.allocate && !(plan->whole_kernel && layout.local_entries)) {
      workspace_slots.push_back(workspace->size());
      workspace->push_back(make_storage(layout));
      workspace->back().setZero();
    } else {
      workspace_slots.push_back(no_value);
    }
  }
  slot_pointers.resize(plan->values.size());
  backend_pointers.resize(plan->value_panel_pointers +
                          plan->external_inputs + plan->outputs);
  factors.reserve(plan->factor_values.size());
  for (size_t i = 0; i < plan->factor_values.size(); ++i)
    factors.push_back(std::make_unique<cached_factor>());
  const size_t solves = std::ranges::count_if(
      plan->operations, [](const auto &operation) {
        return dynamic_cast<const solve_operation *>(operation.get());
      });
  solve_rhs_buffers.resize(solves);
  solve_output_buffers.resize(solves);
  if (plan->whole_kernel)
    whole_kernel_state = plan->whole_kernel(0, nullptr, 0, nullptr);
}

casadi_mx_graph_instance::~casadi_mx_graph_instance() {
  if (plan && plan->whole_kernel && whole_kernel_state)
    plan->whole_kernel(1, whole_kernel_state, 0, nullptr);
}

void casadi_mx_graph_instance::run(size_t entry,
                                   std::span<scalar_t *> pointers) const {
  if (entry >= plan->entry_outputs.size())
    throw std::out_of_range("CasADi MX graph entry is out of range");
  const auto bind_panels = [](const sparse_matrix &matrix,
                              std::vector<scalar_t *> &bound) {
    bound.clear();
    const size_t diagonal = matrix.diagonal_segments_.empty()
                                ? matrix.diag_panels_.size()
                                : matrix.diagonal_segments_.size();
    bound.reserve(matrix.dense_panels_.size() + diagonal +
                  matrix.eye_panels_.size());
    for (const auto &panel : matrix.dense_panels_)
      bound.push_back(const_cast<scalar_t *>(panel.data_.data()));
    if (matrix.diagonal_segments_.empty()) {
      for (const auto &panel : matrix.diag_panels_)
        bound.push_back(const_cast<scalar_t *>(panel.data_.data()));
    } else {
      for (const auto &segment : matrix.diagonal_segments_)
        bound.push_back(const_cast<scalar_t *>(
            matrix.diag_panels_[segment.storage_panel].data_.data()) +
                        segment.storage_offset);
    }
    for (const auto &panel : matrix.eye_panels_)
      bound.push_back(const_cast<scalar_t *>(panel.data_.data()));
  };
  const bool bind_external =
      bound_external.size() != pointers.size() ||
      !std::equal(bound_external.begin(), bound_external.end(),
                  pointers.begin());
  const bool bind_workspace = !workspace_bound;
  if (bind_workspace) {
    // All graph instances sharing lag_data finish construction before the
    // first approximation update.  Bind their persistent sparse stores once;
    // the runtime schedule then only dispatches panel kernels.
    for (size_t value = 0; value < plan->values.size(); ++value)
      if (workspace_slots[value] != no_value)
        bind_panels(workspace->at(workspace_slots[value]),
                    slot_pointers[value]);
  }
  for (const auto &binding : plan->direct_outputs) {
    if (!plan->values[binding.value].allocate) continue;
    auto &bound = slot_pointers[binding.value];
    bound[binding.panel] =
        pointers[plan->external_inputs + binding.output] + binding.offset;
  }
  if (entry == 0) ++factor_epoch;
  execution_context context{*plan, *this, pointers, entry};
  if (bind_workspace || bind_external) {
    for (const auto &operation : plan->operations)
      if (operation->binding_only()) operation->execute(context);
    for (size_t value = 0; value < plan->values.size(); ++value) {
      const size_t offset = plan->panel_pointer_offsets[value];
      std::copy(slot_pointers[value].begin(), slot_pointers[value].end(),
                backend_pointers.begin() + offset);
    }
    std::copy(pointers.begin(), pointers.end(),
              backend_pointers.begin() + plan->value_panel_pointers);
    workspace_bound = true;
    bound_external.assign(pointers.begin(), pointers.end());
  }
  if (plan->whole_kernel) {
    plan->whole_kernel(2, whole_kernel_state, entry,
                       backend_pointers.data());
  } else {
    for (const operation *operation : plan->entry_schedules[entry])
      operation->execute(context);
  }
  if (std::getenv("MOTO_CHECK_SPARSE_GRAPH")) {
    std::vector<casadi::DM> arguments;
    arguments.reserve(plan->inputs);
    for (size_t input = 0; input < plan->inputs; ++input) {
      std::vector<double> values(plan->reference.nnz_in(input), 0.);
      if (plan->input_values[input] != no_value)
        for (size_t nz = 0; nz < values.size(); ++nz)
          values[nz] = context.read(plan->input_values[input], nz);
      arguments.emplace_back(plan->reference.sparsity_in(input), values,
                             false);
    }
    const auto expected = plan->reference(arguments);
    for (size_t output = 0; output < plan->outputs; ++output) {
      if (plan->output_entries[output] != entry) continue;
      const auto values = expected[output].nonzeros();
      for (size_t nz = 0; nz < values.size(); ++nz) {
        const double actual =
            pointers[plan->external_inputs + output][nz];
        const double error = std::abs(actual - values[nz]);
        if (!(error <= 1e-9 * (1. + std::abs(values[nz])))) {
          std::ostringstream message;
          message << std::setprecision(17)
                  << "sparse graph mismatch: inputs=" << plan->inputs
                  << " outputs=" << plan->outputs << " output=" << output
                  << " nz=" << nz << " actual=" << actual
                  << " expected=" << values[nz] << " error=" << error;
          if (!std::isfinite(values[nz])) {
            message << " inputs:";
            for (size_t input = 0; input < arguments.size(); ++input) {
              const auto data = arguments[input].nonzeros();
              double maximum = 0.;
              size_t finite = 0, nonzero = 0;
              for (const double value : data) {
                if (std::isfinite(value)) ++finite;
                if (value != 0.) ++nonzero;
                if (std::isfinite(value))
                  maximum = std::max(maximum, std::abs(value));
              }
              message << ' ' << plan->reference.name_in(input) << '['
                      << plan->reference.size1_in(input) << 'x'
                      << plan->reference.size2_in(input) << ",nnz="
                      << data.size() << ",live=" << nonzero
                      << ",finite=" << finite << ",max=" << maximum << ']';
            }
          }
          throw std::runtime_error(message.str());
        }
      }
    }
  }
}

size_t casadi_mx_graph_inputs(const casadi_mx_graph_plan &plan) {
  return plan.external_inputs;
}
size_t casadi_mx_graph_outputs(const casadi_mx_graph_plan &plan) {
  return plan.outputs;
}
size_t casadi_mx_graph_entries(const casadi_mx_graph_plan &plan) {
  return plan.entry_outputs.size();
}

} // namespace moto::linear_backend::detail
