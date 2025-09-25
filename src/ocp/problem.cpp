#include <algorithm>
#include <moto/ocp/dynamics.hpp>
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
                    for (const generic_dynamics &dyn : expr_[__dyn]) {
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
        finalized_ = false;
        ex.add_to_ocp_callback(this);
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
ocp_ptr_t ocp::clone(const clone_config &config) const {
    auto prob = std::shared_ptr<ocp>(new ocp(*this));
    prob->finalized_ = false;          // need to reset finalized
    for (auto &p : prob->sub_probs_) { // recursively clone sub problems
        p = p->clone(config);
    } /// @warning this is not guaranteed to be consistent
    auto delete_expr = [&](const expr &ex, bool prune = false) {
        size_t f = ex.field();
        auto &exprs = prob->expr_[f];
        auto &target_list = prune ? prob->pruned_expr_[f] : prob->disabled_expr_[f];
        auto &target_uids = prune ? prob->pruned_uids_ : prob->disabled_uids_;
        target_list.emplace_back(ex);
        target_uids.insert(ex.uid());
        std::erase_if(exprs, [&ex](const shared_expr &e) { return e->uid() == ex.uid(); });
        prob->uids_.erase(ex.uid());
    };
    /// re-enable previously pruned expressions
    auto re_enable_expr = [&](const expr &ex, bool from_pruned = false) {
        size_t f = ex.field();
        auto &exprs = prob->expr_[f];
        auto &target_list = from_pruned ? prob->pruned_expr_[f] : prob->disabled_expr_[f];
        auto &target_uids = from_pruned ? prob->pruned_uids_ : prob->disabled_uids_;
        auto it = std::find_if(target_list.begin(), target_list.end(),
                               [&ex](const shared_expr &e) { return e->uid() == ex.uid(); });
        if (it != target_list.end()) {
            exprs.emplace_back(*it);
            prob->uids_.insert(ex.uid());
            target_list.erase(it);
            target_uids.erase(ex.uid());
            return true;
        }
        return false;
    };
    for (expr &ex : config.activate_list) {
        /// by activating an expression, previously pruned expressions may be re-enabled
        /// but previously user-disabled expressions should not be re-enabled
        if (re_enable_expr(ex, true) || re_enable_expr(ex, false)) {
            // prob->add(ex); // does not previously exist, add as new expr
            throw std::runtime_error(fmt::format("Cannot activate expression {} uid {}, it does not exist in the problem",
                                                 ex.name(), ex.uid()));
        }
    }
    // disable expressions
    for (const expr &ex : config.deactivate_list) {
        delete_expr(ex);
    }
    constexpr auto all_func_fields = concat_fields(func_fields, custom_func_fields);
    // do lazy pruning
    // order: the user-defined deactivate_list -> prune_funcs -> check if any pruned funcs can be re-enabled
    // this should be done iteratively until no more changes (conflict might happen)
    int max_iter = 5;
ITER_START:
    if (max_iter-- == 0) {
        throw std::runtime_error("ocp::clone failed to converge during pruning");
    }
    bool changed = false;
    array_type<std::vector<std::reference_wrapper<const expr>>, all_func_fields> to_delete, to_re_enable;
    for (auto f : all_func_fields) {
        if (prob->pruned_expr_[f].empty())
            continue;
        to_re_enable[f].reserve(prob->pruned_expr_[f].size());
        for (const generic_func &e : prob->pruned_expr_[f]) {
            bool has_active_primal_arg = false;
            for (auto p : primal_fields) {
                if (e.active_num(p, prob.get()) != 0)
                    has_active_primal_arg = true;
            }
            // check if all enable_if_deps are active and all disable_if_deps are inactive
            bool can_re_enable = has_active_primal_arg;
            if (can_re_enable) {
                can_re_enable = e.check_enable(prob.get());
            }
            if (can_re_enable) {
                to_re_enable[f].emplace_back(e);
            }
        }
    }
    // prune
    for (auto f : all_func_fields) {
        if (prob->expr_[f].empty())
            continue;
        to_delete[f].reserve(prob->expr_[f].size());
        for (const generic_func &e : prob->expr_[f]) {
            bool has_active_primal_arg = false;
            for (auto p : primal_fields) {
                if (e.active_num(p, prob.get()) != 0)
                    has_active_primal_arg = true;
            }
            if (!has_active_primal_arg) {
                // prune funcs with no active primal args
                to_delete[f].emplace_back(e);
                // fmt::print("func {} pruned due to no active primal args\n", e.name());
            } else {
                // prune disabled funcs
                if (!e.check_enable(prob.get())){
                    to_delete[f].emplace_back(e);
                    // fmt::print("func {} pruned due to enable/disable conditions\n", e.name());
                }
            }
        }
    }
    // actually delete and re-enable
    for (auto f : all_func_fields) {
        for (const expr &e : to_delete[f]) {
            delete_expr(e, true);
            // fmt::print("func {} pruned\n", e.name());
            changed = true;
        }
        for (const expr &e : to_re_enable[f]) {
            re_enable_expr(e, true);
            // fmt::print("func {} re-enabled\n", e.name());
            changed = true;
        }
    }
    if (changed)
        goto ITER_START;
    return prob;
}
} // namespace moto

template void moto::ocp::add<const moto::shared_expr &>(const moto::shared_expr &ex);
template void moto::ocp::add<moto::shared_expr>(moto::shared_expr &&ex);
template void moto::ocp::add<const moto::shared_expr>(const moto::shared_expr &&ex);