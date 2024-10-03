#ifndef __OPT_SPARSITY__
#define __OPT_SPARSITY__

#include <atri/common/fwd.hpp>

namespace atri {
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
}  // namespace atri

#endif /*__SPARSITY_*/