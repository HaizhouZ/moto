#include <moto/core/linear_backend.hpp>

#include <algorithm>
#include <map>
#include <numeric>
#include <stdexcept>

namespace moto::linear_backend {
namespace {

void hash_combine(size_t &seed, size_t value) {
  seed ^= value + size_t{0x9e3779b9} + (seed << 6) + (seed >> 2);
}

void validate_layout(const matrix_layout &layout) {
  for (const auto &panel : layout.panels) {
    if (!panel.rows || !panel.cols ||
        panel.row_offset + panel.rows > layout.rows ||
        panel.col_offset + panel.cols > layout.cols)
      throw std::invalid_argument("invalid SpMM panel bounds");
    if (panel.pattern != sparsity::dense &&
        panel.pattern != sparsity::diag &&
        panel.pattern != sparsity::eye)
      throw std::invalid_argument("invalid SpMM panel sparsity");
    if (panel.pattern != sparsity::dense && panel.rows != panel.cols)
      throw std::invalid_argument("structured SpMM panel must be square");
  }
}

std::vector<sparse_block_spec>
blocks_from_pattern(const sparse_pattern &pattern) {
  if (!pattern.nnz()) return {};
  if (pattern.nnz() == pattern.rows * pattern.cols)
    return {{0, 0, pattern.rows, pattern.cols, sparsity::dense}};

  const auto contains = [&](size_t row, size_t col) {
    return std::binary_search(pattern.row.begin() + pattern.colind[col],
                              pattern.row.begin() + pattern.colind[col + 1],
                              row);
  };
  bool diagonal = pattern.rows == pattern.cols &&
                  pattern.nnz() == pattern.rows;
  for (size_t i = 0; diagonal && i < pattern.rows; ++i)
    diagonal = contains(i, i);
  if (diagonal)
    return {{0, 0, pattern.rows, pattern.cols, sparsity::diag}};

  std::vector<std::vector<size_t>> by_row(pattern.rows);
  for (size_t col = 0; col < pattern.cols; ++col)
    for (size_t nz = pattern.colind[col]; nz < pattern.colind[col + 1]; ++nz)
      by_row[pattern.row[nz]].push_back(col);

  struct active_rectangle {
    size_t block = 0;
    size_t last_row = 0;
  };
  std::vector<sparse_block_spec> blocks;
  std::map<std::pair<size_t, size_t>, active_rectangle> active;
  for (size_t row = 0; row < pattern.rows; ++row) {
    std::map<std::pair<size_t, size_t>, active_rectangle> next;
    for (size_t i = 0; i < by_row[row].size();) {
      const size_t begin = by_row[row][i];
      size_t end = begin + 1;
      while (++i < by_row[row].size() && by_row[row][i] == end) ++end;
      const auto key = std::pair{begin, end};
      if (const auto found = active.find(key);
          found != active.end() && found->second.last_row + 1 == row) {
        ++blocks[found->second.block].rows;
        next.emplace(key, active_rectangle{found->second.block, row});
      } else {
        blocks.push_back({row, begin, 1, end - begin, sparsity::dense});
        next.emplace(key, active_rectangle{blocks.size() - 1, row});
      }
    }
    active = std::move(next);
  }

  std::map<ptrdiff_t, std::vector<size_t>> diagonal_candidates;
  for (size_t i = 0; i < blocks.size(); ++i)
    if (blocks[i].rows == 1 && blocks[i].cols == 1)
      diagonal_candidates[static_cast<ptrdiff_t>(blocks[i].col) -
                          static_cast<ptrdiff_t>(blocks[i].row)]
          .push_back(i);
  std::vector<bool> consumed(blocks.size());
  std::vector<sparse_block_spec> diagonal_blocks;
  for (auto &[offset, candidates] : diagonal_candidates) {
    (void)offset;
    std::ranges::sort(candidates, {},
                      [&](size_t index) { return blocks[index].row; });
    for (size_t begin = 0; begin < candidates.size();) {
      size_t end = begin + 1;
      while (end < candidates.size() &&
             blocks[candidates[end]].row ==
                 blocks[candidates[end - 1]].row + 1 &&
             blocks[candidates[end]].col ==
                 blocks[candidates[end - 1]].col + 1)
        ++end;
      if (end - begin > 1) {
        const auto &first = blocks[candidates[begin]];
        diagonal_blocks.push_back(
            {first.row, first.col, end - begin, end - begin, sparsity::diag});
        for (size_t i = begin; i < end; ++i)
          consumed[candidates[i]] = true;
      }
      begin = end;
    }
  }
  std::vector<sparse_block_spec> result;
  for (size_t i = 0; i < blocks.size(); ++i)
    if (!consumed[i]) result.push_back(blocks[i]);
  result.insert(result.end(), diagonal_blocks.begin(), diagonal_blocks.end());
  std::ranges::sort(result, {}, [](const sparse_block_spec &block) {
    return std::pair{block.row, block.col};
  });
  return result;
}

struct logical_panel {
  const panel_layout &panel;
  bool transpose = false;

  size_t row() const {
    return transpose ? panel.col_offset : panel.row_offset;
  }
  size_t col() const {
    return transpose ? panel.row_offset : panel.col_offset;
  }
  size_t rows() const { return transpose ? panel.cols : panel.rows; }
  size_t cols() const { return transpose ? panel.rows : panel.cols; }
  bool structured() const { return panel.pattern != sparsity::dense; }
};

bool contains(const panel_layout &panel, const sparse_block_spec &block) {
  if (panel.pattern == sparsity::dense)
    return panel.row_offset <= block.row && panel.col_offset <= block.col &&
           block.row + block.rows <= panel.row_offset + panel.rows &&
           block.col + block.cols <= panel.col_offset + panel.cols;
  return block.pattern == sparsity::diag &&
         panel.pattern == sparsity::diag &&
         panel.row_offset <= block.row && panel.col_offset <= block.col &&
         block.row - panel.row_offset == block.col - panel.col_offset &&
         block.row + block.rows <= panel.row_offset + panel.rows &&
         block.col + block.cols <= panel.col_offset + panel.cols;
}

bool support_overlaps(const panel_layout &panel,
                      const panel_layout &dense) {
  if (panel.pattern == sparsity::dense)
    return panel.row_offset < dense.row_offset + dense.rows &&
           dense.row_offset < panel.row_offset + panel.rows &&
           panel.col_offset < dense.col_offset + dense.cols &&
           dense.col_offset < panel.col_offset + panel.cols;
  const size_t begin = std::max(
      {size_t{0},
       dense.row_offset > panel.row_offset
           ? dense.row_offset - panel.row_offset
           : size_t{0},
       dense.col_offset > panel.col_offset
           ? dense.col_offset - panel.col_offset
           : size_t{0}});
  const size_t end = std::min(
      {panel.rows, dense.row_offset + dense.rows > panel.row_offset
                       ? dense.row_offset + dense.rows - panel.row_offset
                       : size_t{0},
       dense.col_offset + dense.cols > panel.col_offset
           ? dense.col_offset + dense.cols - panel.col_offset
           : size_t{0}});
  return begin < end;
}

void ensure_kernel_coverage(matrix_layout &layout,
                            std::span<const spmm_panel_product> products) {
  for (const auto &product : products) {
    const auto &block = product.output;
    if (std::ranges::any_of(layout.panels,
                            [&](const auto &panel) {
                              return contains(panel, block);
                            }))
      continue;
    panel_layout merged{sparsity::dense, block.row, block.col, block.rows,
                        block.cols};
    bool changed;
    do {
      changed = false;
      for (size_t i = 0; i < layout.panels.size();) {
        const auto panel = layout.panels[i];
        if (!support_overlaps(panel, merged)) {
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
        layout.panels.erase(layout.panels.begin() + i);
        changed = true;
      }
    } while (changed);
    layout.panels.push_back(merged);
  }
  std::ranges::stable_sort(layout.panels, {}, [](const panel_layout &panel) {
    return panel.pattern == sparsity::dense ? 0
           : panel.pattern == sparsity::diag ? 1
                                             : 2;
  });
}

sparse_pattern operand_pattern(spmm_operand &operand) {
  sparse_pattern pattern;
  if (operand.pattern.colind.empty()) {
    pattern = analyze_pattern(operand.layout);
    operand.pattern = pattern;
  } else {
    operand.pattern.validate();
    if (operand.pattern.rows != operand.layout.rows ||
        operand.pattern.cols != operand.layout.cols)
      throw std::invalid_argument("SpMM operand pattern shape mismatch");
    pattern = operand.pattern;
  }
  return operand.transpose ? transpose_pattern(pattern) : pattern;
}

} // namespace

void sparse_pattern::validate() const {
  if (colind.size() != cols + 1 || colind.empty() || colind.front() != 0 ||
      colind.back() != row.size())
    throw std::invalid_argument("invalid sparse pattern column pointers");
  for (size_t col = 0; col < cols; ++col) {
    if (colind[col] > colind[col + 1])
      throw std::invalid_argument("unordered sparse pattern columns");
    size_t previous = 0;
    bool first = true;
    for (size_t nz = colind[col]; nz < colind[col + 1]; ++nz) {
      if (row[nz] >= rows || (!first && row[nz] <= previous))
        throw std::invalid_argument("invalid sparse pattern row index");
      first = false;
      previous = row[nz];
    }
  }
}

size_t sparse_pattern_hash::operator()(const sparse_pattern &pattern) const
    noexcept {
  size_t seed = pattern.rows;
  hash_combine(seed, pattern.cols);
  for (const size_t value : pattern.colind) hash_combine(seed, value);
  for (const size_t value : pattern.row) hash_combine(seed, value);
  return seed;
}

sparse_pattern analyze_pattern(const matrix_layout &layout, bool transpose) {
  validate_layout(layout);
  std::vector<std::vector<size_t>> columns(layout.cols);
  for (const auto &panel : layout.panels) {
    if (panel.pattern == sparsity::dense) {
      for (size_t col = panel.col_offset;
           col < panel.col_offset + panel.cols; ++col)
        for (size_t row = panel.row_offset;
             row < panel.row_offset + panel.rows; ++row)
          columns[col].push_back(row);
    } else {
      for (size_t i = 0; i < panel.rows; ++i)
        columns[panel.col_offset + i].push_back(panel.row_offset + i);
    }
  }
  sparse_pattern pattern{layout.rows, layout.cols, {0}, {}};
  for (auto &column : columns) {
    std::ranges::sort(column);
    column.erase(std::unique(column.begin(), column.end()), column.end());
    pattern.row.insert(pattern.row.end(), column.begin(), column.end());
    pattern.colind.push_back(pattern.row.size());
  }
  return transpose ? transpose_pattern(pattern) : pattern;
}

sparse_pattern transpose_pattern(const sparse_pattern &pattern) {
  pattern.validate();
  sparse_pattern result{pattern.cols, pattern.rows,
                        std::vector<size_t>(pattern.rows + 1), {}};
  for (const size_t row : pattern.row) ++result.colind[row + 1];
  std::partial_sum(result.colind.begin(), result.colind.end(),
                   result.colind.begin());
  result.row.resize(pattern.nnz());
  auto next = result.colind;
  for (size_t col = 0; col < pattern.cols; ++col)
    for (size_t nz = pattern.colind[col]; nz < pattern.colind[col + 1]; ++nz)
      result.row[next[pattern.row[nz]]++] = col;
  return result;
}

sparse_pattern analyze_spmm_pattern(const sparse_pattern &lhs,
                                    const sparse_pattern &rhs) {
  lhs.validate();
  rhs.validate();
  if (lhs.cols != rhs.rows)
    throw std::invalid_argument("SpMM pattern dimension mismatch");
  sparse_pattern output{lhs.rows, rhs.cols, {0}, {}};
  std::vector<size_t> marker(lhs.rows, 0);
  size_t epoch = 0;
  std::vector<size_t> column;
  for (size_t col = 0; col < rhs.cols; ++col) {
    column.clear();
    if (++epoch == 0) {
      std::ranges::fill(marker, 0);
      ++epoch;
    }
    for (size_t rnz = rhs.colind[col]; rnz < rhs.colind[col + 1]; ++rnz) {
      const size_t inner = rhs.row[rnz];
      for (size_t lnz = lhs.colind[inner]; lnz < lhs.colind[inner + 1];
           ++lnz) {
        const size_t row = lhs.row[lnz];
        if (marker[row] == epoch) continue;
        marker[row] = epoch;
        column.push_back(row);
      }
    }
    std::ranges::sort(column);
    output.row.insert(output.row.end(), column.begin(), column.end());
    output.colind.push_back(output.row.size());
  }
  return output;
}

matrix_layout panelize_pattern(const sparse_pattern &pattern) {
  pattern.validate();
  matrix_layout result{pattern.rows, pattern.cols, {}};
  const auto plan = make_sparse_layout_plan(blocks_from_pattern(pattern));
  for (const auto &panel : plan.panels)
    result.panels.push_back({panel.pattern, panel.row, panel.col, panel.rows,
                             panel.cols});
  std::ranges::stable_sort(result.panels, {}, [](const panel_layout &panel) {
    return panel.pattern == sparsity::dense ? 0
           : panel.pattern == sparsity::diag ? 1
                                             : 2;
  });
  return result;
}

size_t spmm_analysis::scalar_products() const {
  return std::accumulate(products.begin(), products.end(), size_t{0},
                         [](size_t total, const auto &product) {
                           return total + product.scalar_products;
                         });
}

spmm_analysis analyze_spmm(spmm_operand lhs, spmm_operand rhs) {
  if (lhs.cols() != rhs.rows())
    throw std::invalid_argument("SpMM operand dimension mismatch");
  const sparse_pattern lhs_pattern = operand_pattern(lhs);
  const sparse_pattern rhs_pattern = operand_pattern(rhs);
  spmm_analysis result{std::move(lhs), std::move(rhs),
                       analyze_spmm_pattern(lhs_pattern, rhs_pattern), {}, {}};
  result.output_layout = panelize_pattern(result.output_pattern);
  for (size_t li = 0; li < result.lhs.layout.panels.size(); ++li) {
    const logical_panel l{result.lhs.layout.panels[li], result.lhs.transpose};
    for (size_t ri = 0; ri < result.rhs.layout.panels.size(); ++ri) {
      const logical_panel r{result.rhs.layout.panels[ri],
                            result.rhs.transpose};
      const size_t begin = std::max(l.col(), r.row());
      const size_t end = std::min(l.col() + l.cols(), r.row() + r.rows());
      if (end <= begin) continue;
      const size_t reduction = end - begin;
      const size_t rows = l.structured() ? reduction : l.rows();
      const size_t cols = r.structured() ? reduction : r.cols();
      const size_t output_row =
          l.row() + (l.structured() ? begin - l.col() : 0);
      const size_t output_col =
          r.col() + (r.structured() ? begin - r.row() : 0);
      const bool diagonal = l.structured() && r.structured();
      const size_t products = diagonal ? reduction
                              : l.structured() ? reduction * r.cols()
                              : r.structured() ? l.rows() * reduction
                                               : l.rows() * r.cols() * reduction;
      result.products.push_back(
          {li, ri, begin, reduction,
           {output_row, output_col, rows, cols,
            diagonal ? sparsity::diag : sparsity::dense},
           products});
    }
  }
  ensure_kernel_coverage(result.output_layout, result.products);
  return result;
}

} // namespace moto::linear_backend
