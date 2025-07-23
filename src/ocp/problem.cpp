#include <moto/ocp/problem.hpp>
#include <moto/ocp/sym.hpp>
namespace moto {
INIT_UID_(ocp);
bool ocp::add_impl(expr &ex) {
    size_t _uid = ex.uid();
    if (d_idx_.find(_uid) == d_idx_.end()) { // skip repeated
        // add dependencies
        auto &dep = ex.dep();
        if (!dep.empty()) {
            add(dep);
        }
        if (!ex.finalize()) {
            throw std::runtime_error(fmt::format("cannot finalize expr {} uid {}", ex.name(), ex.uid()));
        }
        size_t &n0 = dim_[ex.field()];
        size_t n1 = n0 + ex.dim();
        d_idx_[_uid] = std::make_pair(n0, n1);
        n0 = d_idx_[_uid].second;
        pos_by_uid_.try_emplace(_uid, expr_[ex.field()].size());
        return true;
    }
    return false;
}
} // namespace moto
