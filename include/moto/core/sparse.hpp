#ifndef MOTO_CORE_SPARSE_HPP
#define MOTO_CORE_SPARSE_HPP

#include <moto/core/fwd.hpp>

namespace moto {
enum class sparsity : size_t { dense = 0, diag, eye, num, unknown };
struct sp_info {
  sparsity pattern = sparsity::unknown;
  size_t row_offset = 0;
  size_t col_offset = 0;
  size_t rows = std::numeric_limits<size_t>::max();
  size_t cols = std::numeric_limits<size_t>::max();
};
} // namespace moto

#endif // MOTO_CORE_SPARSE_HPP
