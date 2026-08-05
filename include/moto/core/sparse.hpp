#ifndef MOTO_CORE_SPARSE_HPP
#define MOTO_CORE_SPARSE_HPP

#include <moto/core/fwd.hpp>

#include <span>
#include <vector>

namespace moto {
enum class sparsity : size_t { dense = 0, diag, eye, num, unknown };
struct sp_info {
  sparsity pattern = sparsity::unknown;
  size_t row_offset = 0;
  size_t col_offset = 0;
  size_t rows = std::numeric_limits<size_t>::max();
  size_t cols = std::numeric_limits<size_t>::max();
};
struct indexed_sp_info {
  size_t row_arg = 0;
  size_t col_arg = 0;
  sp_info block;
};
struct sparse_block_spec {
  size_t row = 0, col = 0, rows = 0, cols = 0;
  sparsity pattern = sparsity::unknown;
  bool operator==(const sparse_block_spec &) const = default;
};
enum class sparse_plan_mode : size_t { distinct, additive };
struct sparse_binding_spec {
  sparse_block_spec block;
  sparsity storage_pattern = sparsity::unknown;
  size_t panel = 0, local_row = 0, local_col = 0;
};
struct sparse_layout_plan {
  std::vector<sparse_block_spec> panels;
  std::vector<sparse_binding_spec> bindings;
  bool pack_diagonal_storage = false;
  bool empty() const { return panels.empty(); }
};
sparse_layout_plan
make_sparse_layout_plan(std::span<const sparse_block_spec> blocks,
                        sparse_plan_mode mode = sparse_plan_mode::distinct);
} // namespace moto

#endif // MOTO_CORE_SPARSE_HPP
