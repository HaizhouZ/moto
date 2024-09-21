#ifndef __OPT_SPARSITY__
#define __OPT_SPARSITY__

#include <manbo/common/fwd.hpp>

namespace manbo {
enum class sparsity {
    dense,
    diag,
    single,
};
struct sparsity_info {
    std::vector<sparsity> type_;
    std::vector<double[4]> pos_;
    bool empty() { return type_.empty(); }
};
}  // namespace manbo

#endif /*__SPARSITY_*/