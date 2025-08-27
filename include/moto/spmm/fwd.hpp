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
} // namespace moto

#endif // MOTO_SPMM_FWD_HPP