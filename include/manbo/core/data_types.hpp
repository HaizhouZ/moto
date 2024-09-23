#ifndef __OCP_DATA_TYPES__
#define __OCP_DATA_TYPES__
#include <manbo/core/expression_base.hpp>
#include <array>

namespace manbo {
typedef std::array<scalar_t*, field::num_sym> primal_data_ptr_t;
}  // namespace manbo

#endif /*__OCP_CHAIN_*/