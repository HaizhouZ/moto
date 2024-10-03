#ifndef __OCP_DATA_TYPES__
#define __OCP_DATA_TYPES__
#include <atri/core/expression_base.hpp>
#include <array>

namespace atri {
typedef std::array<scalar_t*, field::num_sym> primal_data_ptr_t;
}  // namespace atri

#endif /*__OCP_CHAIN_*/