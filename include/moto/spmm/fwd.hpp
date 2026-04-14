#ifndef MOTO_SPMM_FWD_HPP
#define MOTO_SPMM_FWD_HPP

#include <moto/core/fwd.hpp>

namespace moto {
enum class sparsity : size_t {
    dense = 0,
    diag,
    eye,
    num,
    unknown
};
struct sp_info {
    sparsity pattern = sparsity::unknown;
    size_t row_offset = 0;
    size_t col_offset = 0;
    size_t rows = std::numeric_limits<size_t>::max();
    size_t cols = std::numeric_limits<size_t>::max();
};
namespace spmm {
struct clip_info {
    constexpr static size_t no_clip = std::numeric_limits<size_t>::max();
    size_t lhs_outer_st = 0;
    size_t lhs_outer_end = no_clip;
    size_t rhs_outer_st = 0;
    size_t rhs_outer_end = no_clip;
    size_t inner_st = 0;
    size_t inner_end = no_clip;
};
} // namespace spmm
} // namespace moto

#endif // MOTO_SPMM_FWD_HPP