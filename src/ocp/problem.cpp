#include <moto/ocp/problem.hpp>

namespace moto {
INIT_UID_(ocp);
bool ocp::add_impl(expr &ex) {
    size_t _uid = ex.uid();
    if (d_idx_.find(_uid) == d_idx_.end()) { // skip repeated
        // add dependencies
        if (!ex.finalize()) {
            throw std::runtime_error(fmt::format("cannot finalize expr {} uid {}", ex.name(), ex.uid()));
        }
        const auto &dep = ex.dep();
        if (!dep.empty()) {
            add(dep);
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
void ocp::wait_until_ready() const {
    for (const auto &f : expr_) {
        for (const auto &e : f) {
            if (!e->wait_until_ready()) {
                throw std::runtime_error(fmt::format("Expression {} with uid {} failed to be ready",
                                                     e->name(), e->uid()));
            }
        }
    }
}
} // namespace moto

template void moto::ocp::add<const moto::shared_expr &>(const moto::shared_expr &ex);
template void moto::ocp::add<moto::shared_expr>(moto::shared_expr &&ex);
template void moto::ocp::add<const moto::shared_expr>(const moto::shared_expr &&ex);