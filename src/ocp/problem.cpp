#include <algorithm>
#include <moto/ocp/impl/func.hpp>
#include <moto/ocp/problem.hpp>

namespace moto {
INIT_UID_(ocp);
bool ocp::add_impl(expr &ex) {
    size_t _uid = ex.uid();
    if (!uids_.contains(_uid)) { // skip repeated
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
        uids_.insert(_uid);
        return true;
    }
    return false;
}
void ocp::finalize() {
    static std::mutex finalize_mutex_;
    std::lock_guard lock(finalize_mutex_);
    if (!finalized_) {
        this->set_dim_and_idx();
        this->finalized_ = true;
    }
}
void ocp::set_dim_and_idx() {
    for (size_t i = 0; i < field::num; i++) {
        dim_[i] = 0;
        if (i < field::num_prim)
            tdim_[i] = 0;
        size_t cur = 0, idx = 0, tcur = 0;
        for (expr &ex : expr_[i]) {
            dim_[i] += ex.dim();
            d_idx_[ex.uid()] = cur;
            cur += ex.dim();
            if (i < field::num_prim) {
                tdim_[i] += ex.tdim();
                d_idx_tangent_[ex.uid()] = tcur;
                tcur += ex.tdim();
            }
            pos_by_uid_[ex.uid()] = idx++;
        }
    }
}
void ocp::maintain_order(expr &ex) {
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
                throw std::runtime_error(fmt::format("order maintenance failure: field {} has exprs not in dynamics args", field::name(f)));
                tmp.insert(tmp.end(), exprs.begin(), exprs.end());
            }
            exprs = std::move(tmp);
        }
    }
}
void ocp::print_summary() {
    finalize();
    fmt::print("-------------------------------------------------\n");
    fmt::print("problem uid {}\n", uid_);
    for (size_t i = 0; i < field::num; i++) {
        if (exprs(i).size() > 0) {
            fmt::print("field : {}, \ttotal dim {}\n",
                       field::name(static_cast<field_t>(i)),
                       dim(i));
            for (const auto &expr : exprs(i)) {
                fmt::print(" - {} uid {} dim: {}\n", expr->name(), expr->uid(), expr->dim());
            }
        }
    }
    fmt::print("-------------------------------------------------\n");
}
void ocp::wait_until_ready() {
    for (const auto &f : expr_) {
        for (const auto &e : f) {
            if (!e->wait_until_ready()) {
                throw std::runtime_error(fmt::format("Expression {} with uid {} failed to be ready",
                                                     e->name(), e->uid()));
            }
        }
    }
    finalize();
}
ocp_ptr_t ocp::clone(const expr_inarg_list &deactivate_list) const {
    auto prob = std::shared_ptr<ocp>(new ocp(*this));
    prob->finalized_ = false; // need to reset finalized
    // disable expressions
    for (const expr &ex : deactivate_list) {
        size_t f = ex.field();
        auto &exprs = prob->expr_[f];
        std::erase_if(exprs, [&ex](const shared_expr &e) { return e->uid() == ex.uid(); });
        prob->uids_.erase(ex.uid());
    }
    return prob;
}
} // namespace moto

template void moto::ocp::add<const moto::shared_expr &>(const moto::shared_expr &ex);
template void moto::ocp::add<moto::shared_expr>(moto::shared_expr &&ex);
template void moto::ocp::add<const moto::shared_expr>(const moto::shared_expr &&ex);