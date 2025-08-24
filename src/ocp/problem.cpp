#include <algorithm>
#include <moto/ocp/impl/func.hpp>
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
        if (!dep.empty()) { // this must be done before currect ex
            // check consistency for dynamics, x and y should not be in dep!
            if (ex.field() == __dyn)
                for (auto f : {__x, __y})
                    for (const generic_func &dyn : expr_[__dyn]) {
                        auto it = std::find_first_of(dep.begin(), dep.end(),
                                                     dyn.in_args(f).begin(), dyn.in_args(f).end());
                        if (it != dep.end()) {
                            throw std::runtime_error(fmt::format("Dynamics {} arg {} uid {} in {} found in {}",
                                                                 ex.name(), (*it)->name(), (*it)->uid(), f, dyn.name()));
                        }
                    }
            add(dep);
        }
        size_t &n0 = dim_[ex.field()];
        size_t n1 = n0 + ex.dim();
        d_idx_[_uid] = n0;
        n0 = n1;
        if (ex.field() < field::num_prim) {
            size_t &t0 = tdim_[ex.field()];
            size_t t1 = t0 + ex.tdim();
            d_idx_tangent_[_uid] = t0;
            t0 = t1;
        }
        pos_by_uid_.try_emplace(_uid, num(ex.field()));
        return true;
    }
    return false;
}
void ocp::maintain_order(expr &ex) {
    auto update_pos = [this](const expr_list &exprs) {
        size_t cur = 0, idx = 0, tcur = 0;
        for (expr &ex : exprs) {
            d_idx_[ex.uid()] = cur;
            d_idx_tangent_[ex.uid()] = tcur;
            cur += ex.dim();
            tcur += ex.tdim();
            pos_by_uid_[ex.uid()] = idx++;
        }
    };
    if (ex.field() == __dyn) {
        auto &dyns = exprs(__dyn);
        for (auto f : {__x, __y}) {
            expr_list tmp;
            auto &exprs = expr_[f];
            tmp.reserve(exprs.size());
            for (const generic_func &dyn : dyns) {
                for (const expr &arg : dyn.in_args(f)) {
                    auto it = std::find(exprs.begin(), exprs.end(), arg);
                    if (it == exprs.end())
                        throw std::runtime_error(fmt::format("order maintenance failure: Dynamics {} arg {} uid {} not found in field {}",
                                                             dyn.name(), arg.name(), arg.uid(), f));
                    tmp.emplace_back(std::move(*it));
                }
            }
            std::erase_if(exprs, [&tmp](const shared_expr &e) { return !bool(e); });
            if (!exprs.empty()) {
                tmp.insert(tmp.end(), exprs.begin(), exprs.end());
            }
            exprs = std::move(tmp);
            update_pos(exprs);
        }
    }
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