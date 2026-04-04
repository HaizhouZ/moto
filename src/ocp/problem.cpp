#include <algorithm>
#include <cstdlib>
#include <moto/ocp/constr.hpp>
#include <moto/ocp/cost.hpp>
#include <moto/ocp/dynamics.hpp>
#include <moto/ocp/problem.hpp>

namespace moto {
INIT_UID_(ocp_base);

namespace {
bool compose_trace_enabled() {
    static const bool enabled = std::getenv("MOTO_TRACE_COMPOSE") != nullptr;
    return enabled;
}

bool is_path_state_term(const shared_expr &ex) {
    if (!ex) {
        return false;
    }
    const auto *func = dynamic_cast<const generic_func *>(ex.get());
    if (func == nullptr) {
        return false;
    }
    const bool is_supported_field =
        ex->field() == __eq_x ||
        ex->field() == __eq_x_soft ||
        ex->field() == __undefined;
    if (!is_supported_field) {
        return false;
    }
    bool has_x = false;
    for (const sym &arg : func->in_args()) {
        if (arg.field() == __u || arg.field() == __y) {
            return false;
        }
        if (arg.field() == __x) {
            has_x = true;
        }
    }
    return has_x;
}

bool is_pure_state_cost_term(const shared_expr &ex) {
    if (!ex || ex->field() != __cost) {
        return false;
    }
    const auto *cost_expr = dynamic_cast<const generic_cost *>(ex.get());
    if (cost_expr != nullptr && cost_expr->terminal_add()) {
        return false;
    }
    const auto *func = dynamic_cast<const generic_func *>(ex.get());
    if (func == nullptr) {
        return false;
    }
    bool has_x = false;
    for (const sym &arg : func->in_args()) {
        if (arg.field() == __u || arg.field() == __y) {
            return false;
        }
        if (arg.field() == __x) {
            has_x = true;
        }
    }
    return has_x;
}

shared_expr lower_node_term_for_edge(const shared_expr &ex, const ocp_base &prob) {
    auto lowered = ex.clone();
    auto *func = dynamic_cast<generic_func *>(lowered.get());
    if (func == nullptr) {
        return lowered;
    }
    auto args = func->in_args();
    for (const sym &arg : args) {
        if (arg.field() == __x) {
            if (compose_trace_enabled()) {
                fmt::print("lowering node term {} in composed ocp uid {} during edge compose: {} -> {} (x_k -> y_k storage)\n",
                           ex->name(), prob.uid(), arg.name(), arg.next()->name());
            }
            func->substitute_argument(arg, arg.next());
        }
    }
    return lowered;
}

void append_node_terms(const node_ocp_ptr_t &node_prob,
                       const edge_ocp_ptr_t &edge_prob,
                       bool lower_path_state_terms,
                       bool skip_path_state_terms,
                       bool only_path_state_terms = false) {
    if (!node_prob) {
        return;
    }
    for (size_t f = 0; f < field::num; ++f) {
        for (const shared_expr &expr : node_prob->exprs(f)) {
            const bool is_path_state = is_path_state_term(expr);
            const bool is_pure_state_cost = is_pure_state_cost_term(expr);
            const bool is_lowerable_node_term = is_path_state || is_pure_state_cost;
            if (only_path_state_terms && !is_lowerable_node_term) {
                continue;
            }
            if (is_lowerable_node_term) {
                if (skip_path_state_terms) {
                    continue;
                }
                if (lower_path_state_terms) {
                    edge_prob->add(lower_node_term_for_edge(expr, *edge_prob));
                    continue;
                }
            }
            edge_prob->add(expr);
        }
    }
}
} // namespace

bool ocp_base::add_impl(expr &ex) {
    return add_impl(shared_expr(ex));
}
bool ocp_base::add_impl(shared_expr ex, bool terminal) {
    std::string reason;
    if (!accepts_term(ex, terminal, &reason)) {
        if (reason.empty()) {
            reason = "expression is incompatible with this problem type";
        }
        throw std::runtime_error(fmt::format(
            "Cannot add expression {} uid {} to problem uid {}: {}",
            ex->name(), ex->uid(), uid_, reason));
    }
    ex->prepare_add_to_ocp(terminal);
    size_t _uid = ex->uid();
    if (!contains(*ex, false)) { // skip repeated in the current problem only
        auto *ex_ptr = ex.get();
        ex_ptr->add_to_ocp_callback(this);
        // add dependencies
        if (!ex->finalize()) {
            throw std::runtime_error(fmt::format("cannot finalize expr {} uid {}", ex->name(), ex->uid()));
        }
        const auto &dep = ex->dep();
        if (!dep.empty()) { // this must be done before currect ex
            // check consistency for dynamics, x and y should not be in dep!
            if (ex->field() == __dyn && !allow_inconsistent_dynamics_)
                for (auto f : {__x, __y})
                    for (const generic_dynamics &dyn : expr_[__dyn]) {
                        auto it = std::ranges::find_if(dep, [&](const sym &s) { return dyn.has_arg(s); });
                        if (it != dep.end()) {
                            throw std::runtime_error(
                                fmt::format("Dynamics {} arg {} uid {} in {} found in dynamics {}. "
                                            "Overlapping state variables in dynamics is not allowed to avoid inconsistency."
                                            " If you want to allow this, set allow_inconsistent_dynamics to true.",
                                            ex->name(),
                                            (*it)->name(), (*it)->uid(), f, dyn.name()));
                        }
                    }
            for (expr &arg : dep) {
                if (!contains(arg, false)) {
                    add_impl(arg);
                }
            }
        }
        finalized_ = false; // need to reset finalized to allow adding more expr, will be set to true in finalize()
        if (ex->default_active_status()) {
            uids_.insert(_uid);
            expr_[ex->field()].emplace_back(std::move(ex));
        } else { // add to disabled list
            disabled_uids_.insert(_uid);
            disabled_expr_[ex->field()].emplace_back(std::move(ex));
        }
        return true;
    }
    return false;
}
bool ocp_base::add_terminal_impl(expr &ex) {
    if (ex.finalized()) {
        auto ex_terminal = shared_expr(ex).clone();
        return add_impl(std::move(ex_terminal), true);
    }
    return add_impl(shared_expr(ex), true);
}
bool ocp_base::contains(const expr &ex, bool include_sub_prob) const {
    return uids_.contains(ex.uid()) ||
           disabled_uids_.contains(ex.uid()) ||
           pruned_uids_.contains(ex.uid()) ||
           (include_sub_prob &&
            std::any_of(sub_probs_.begin(),
                        sub_probs_.end(),
                        [&](const ocp_base_ptr_t &p) { return p->contains(ex, true); }));
}
bool ocp_base::is_active(const expr &ex, bool include_sub_prob) const {
    return uids_.contains(ex.uid()) ||
           (include_sub_prob &&
            std::any_of(sub_probs_.begin(),
                        sub_probs_.end(),
                        [&](const ocp_base_ptr_t &p) { return p->is_active(ex, true); }));
}
void ocp_base::finalize() {
    static std::mutex finalize_mutex_;
    std::lock_guard lock(finalize_mutex_);
    if (!finalized_) {
        compiled_projector_layout_.clear();
        if (apply_projector_layout_ && !projector_layout_.empty()) {
            compile_projector_layout();
        }
        if (expr_[__dyn].size() > 0 && automatic_reorder_primal_)
            maintain_order();
        this->set_dim_and_idx();
        this->finalized_ = true;
    }
}
void ocp_base::refresh_after_clone(const active_status_config &config) {
    finalized_ = false;
    for (auto &p : sub_probs_) {
        p = p->clone_base(config);
    }
    if (!config.empty()) {
        update_active_status(config, false);
    }
}
void ocp_base::set_dim_and_idx() {
    for (size_t i = 0; i < field::num; i++) {
        dim_[i] = 0;
        if (i < field::num_prim)
            tdim_[i] = 0; // only primal fields have tangent space dimension
        size_t cur = 0, idx = 0, tcur = 0;
        // assign data index and position for each expression,
        // also calculate total dimension for each field
        for (const expr &ex : expr_[i]) {
            dim_[i] += ex.dim();
            flatten_idx_[ex.uid()] = cur;
            cur += ex.dim();
            if (i < field::num_prim) {
                tdim_[i] += ex.tdim();
                flatten_tidx_[ex.uid()] = tcur;
                tcur += ex.tdim();
            }
            pos_by_uid_[ex.uid()] = idx++;
        }
    }
}

void ocp_base::maintain_order() {
    expr_list tmp;
    for (auto f : {__x, __y}) {
        auto &syms = expr_[f];
        tmp.reserve(syms.size());
        for (const generic_func &dyn : exprs(__dyn)) {
            for (const expr &arg : dyn.in_args(f)) {
                auto it = std::find(syms.begin(), syms.end(), arg);
                if (it == syms.end())
                    throw std::runtime_error(fmt::format(
                        "order maintenance failure: "
                        "Dynamics {} arg {} uid {} not found in field {}",
                        dyn.name(), arg.name(), arg.uid(), f));
                tmp.emplace_back(std::move(*it));
            }
        }
        // remove duplicates moved to tmp
        std::erase_if(syms, [&](auto &&e) { return !e; });
        if (!syms.empty()) {
            // state variables not belonging to any dynamics are forbidden
            throw std::runtime_error(fmt::format(
                "order maintenance failure: "
                " field {} has exprs not in dynamics args",
                f));
        }
        syms.swap(tmp); // now syms is in the order of dynamics args
    }
}
void ocp_base::print_summary() {
    finalize();
    fmt::print("-------------------------------------------------\n");
    fmt::print("problem uid {}\n", uid_);
    for (size_t i = 0; i < field::num; i++) {
        if (exprs(i).size() > 0) {
            fmt::print("field : {}, \ttotal dim {}, \ttotal tdim {}\n",
                       static_cast<field_t>(i),
                       dim(i), i < field::num_prim ? tdim(i) : 0);
            for (const auto &expr : exprs(i)) {
                fmt::print(" - {} uid {} dim: {} tdim: {}\n",
                           expr->name(), expr->uid(), expr->dim(), expr->tdim());
            }
        }
    }
    fmt::print("-------------------------------------------------\n");
}
void ocp_base::wait_until_ready() {
    for (const auto &f : expr_) {
        for (const auto &e : f) {
            if (!e->wait_until_ready()) {
                throw std::runtime_error(fmt::format(
                    "Expression {} with uid {} failed to be ready",
                    e->name(), e->uid()));
            }
        }
    }
    finalize();
}
bool ocp_base::accepts_term(const shared_expr &ex, bool terminal, std::string *reason) const {
    static_cast<void>(ex);
    static_cast<void>(terminal);
    if (reason != nullptr) {
        reason->clear();
    }
    return true;
}

ocp_ptr_t ocp::clone(const active_status_config &config) const {
    auto prob = ocp_ptr_t(new ocp(*this));
    prob->refresh_after_clone(config);
    return prob;
}
node_ocp_ptr_t node_ocp::compose(const node_ocp_ptr_t &base_prob) {
    return base_prob ? base_prob->clone_node() : node_ocp::create();
}
node_ocp_ptr_t node_ocp::clone_node(const active_status_config &config) const {
    auto prob = node_ocp_ptr_t(new node_ocp(*this));
    prob->refresh_after_clone(config);
    return prob;
}
bool node_ocp::accepts_term(const shared_expr &ex, bool terminal, std::string *reason) const {
    static_cast<void>(terminal);
    if (!ex) {
        if (reason != nullptr) {
            *reason = "null expression";
        }
        return false;
    }
    if (ex->field() == __dyn) {
        if (reason != nullptr) {
            *reason = "node_ocp only accepts node-local terms; dynamics must be added to an edge_ocp";
        }
        return false;
    }
    const auto *func = dynamic_cast<const generic_func *>(ex.get());
    if (func == nullptr) {
        return true;
    }
    for (const sym &arg : func->in_args()) {
        if (arg.field() == __y) {
            if (reason != nullptr) {
                *reason = fmt::format(
                    "node_ocp terms may only depend on x/u/p-style node variables; found y argument {}",
                    arg.name());
            }
            return false;
        }
    }
    return true;
}
edge_ocp_ptr_t edge_ocp::clone_edge(const active_status_config &config) const {
    auto prob = edge_ocp_ptr_t(new edge_ocp(*this));
    prob->refresh_after_clone(config);
    return prob;
}
edge_ocp_ptr_t edge_ocp::compose(const node_ocp_ptr_t &st_node_prob,
                                 const edge_ocp_ptr_t &edge_prob,
                                 const node_ocp_ptr_t &lowered_node_prob,
                                 bool skip_st_path_state_terms) {
    auto prob = edge_prob ? edge_prob->clone_edge() : edge_ocp::create();
    prob->__set_apply_projector_layout(true);
    prob->bind_nodes(st_node_prob, edge_prob ? edge_prob->ed_node_prob() : node_ocp_ptr_t{});
    if (st_node_prob && edge_prob) {
        active_status_config config;
        for (auto primal_field : primal_fields) {
            for (const shared_expr &expr : edge_prob->exprs(primal_field)) {
                if (!st_node_prob->contains(*expr, false) || st_node_prob->is_active(*expr, false)) {
                    continue;
                }
                config.deactivate_list.emplace_back(*expr);
            }
        }
        if (!config.empty()) {
            prob->update_active_status(config, false);
        }
    }
    append_node_terms(st_node_prob, prob, true, skip_st_path_state_terms);
    append_node_terms(lowered_node_prob, prob, true, false, true);
    if (st_node_prob) {
        prob->merge_projector_layout_from(*st_node_prob);
    }
    if (lowered_node_prob) {
        prob->merge_projector_layout_from(*lowered_node_prob);
    }
    return prob;
}
void edge_ocp::bind_nodes(const node_ocp_ptr_t &st, const node_ocp_ptr_t &ed) {
    st_node_prob_ = st;
    ed_node_prob_ = ed;
}
edge_ocp_ptr_t edge_ocp::compose() const {
    return compose(st_node_prob_, clone_edge());
}
void ocp_base::update_active_status(const active_status_config &config, bool update_sub_probs) {
    if (update_sub_probs) {
        for (auto &p : sub_probs_) {
            p->update_active_status(config, true);
        }
    }
    auto delete_expr = [&](const expr &ex, bool prune = false) {
        size_t f = ex.field();
        auto &exprs = expr_[f];
        auto &target_list = prune ? pruned_expr_[f] : disabled_expr_[f];
        auto &target_uids = prune ? pruned_uids_ : disabled_uids_;
        auto it = std::find_if(exprs.begin(), exprs.end(),
                               [&ex](const shared_expr &e) { return e->uid() == ex.uid(); });
        if (it == exprs.end()) {
            if (std::ranges::any_of(sub_probs_, [&](const ocp_base_ptr_t &p) {
                    return p->contains(ex, true);
                })) {
                return;
            }
            throw std::runtime_error(fmt::format(
                "Cannot deactivate expression {} uid {}, it does not exist in the active problem",
                ex.name(), ex.uid()));
        }
        target_list.emplace_back(*it);
        target_uids.insert(ex.uid());
        exprs.erase(it);
        uids_.erase(ex.uid());
    };
    /// re-enable previously pruned expressions
    auto re_enable_expr = [&](const expr &ex, bool from_pruned = false) {
        size_t f = ex.field();
        auto &exprs = expr_[f];
        auto &target_list = from_pruned ? pruned_expr_[f] : disabled_expr_[f];
        auto &target_uids = from_pruned ? pruned_uids_ : disabled_uids_;
        auto it = std::find(target_list.begin(), target_list.end(), ex);
        if (it != target_list.end()) {
            exprs.emplace_back(*it);
            uids_.insert(ex.uid());
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
            // add(ex); // does not previously exist, add as new expr
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
        if (pruned_expr_[f].empty())
            continue;
        to_re_enable[f].reserve(pruned_expr_[f].size());
        for (const generic_func &e : pruned_expr_[f]) {
            bool has_active_primal_arg = false;
            for (auto p : primal_fields) {
                if (e.active_num(p, this) != 0)
                    has_active_primal_arg = true;
            }
            // check if all enable_if_deps are active and all disable_if_deps are inactive
            bool can_re_enable = has_active_primal_arg;
            if (can_re_enable) {
                can_re_enable = e.check_enable(this);
            }
            if (can_re_enable) {
                to_re_enable[f].emplace_back(e);
            }
        }
    }
    // prune
    for (auto f : all_func_fields) {
        if (expr_[f].empty())
            continue;
        to_delete[f].reserve(expr_[f].size());
        for (const generic_func &e : expr_[f]) {
            bool has_active_primal_arg = false;
            for (auto p : primal_fields) {
                if (e.active_num(p, this) != 0)
                    has_active_primal_arg = true;
            }
            if (!has_active_primal_arg) {
                // prune funcs with no active primal args
                to_delete[f].emplace_back(e);
                // fmt::print("func {} pruned due to no active primal args\n", e.name());
            } else {
                // prune disabled funcs
                if (!e.check_enable(this)) {
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
            if (std::find(config.activate_list.begin(), config.activate_list.end(), e) != config.activate_list.end()) {
                throw std::runtime_error(fmt::format("func {} uid {} pruned but also in activate_list",
                                                     e.name(), e.uid()));
            }
            changed = true;
        }
        for (const expr &e : to_re_enable[f]) {
            re_enable_expr(e, true);
            if (std::find(config.deactivate_list.begin(), config.deactivate_list.end(), e) != config.deactivate_list.end()) {
                throw std::runtime_error(fmt::format("func {} uid {} re-enabled but also in deactivate_list",
                                                     e.name(), e.uid()));
            }
            // fmt::print("func {} re-enabled\n", e.name());
            changed = true;
        }
    }
    if (changed)
        goto ITER_START;
    finalized_ = false;
}
} // namespace moto

template void moto::ocp_base::add<const moto::shared_expr &>(const moto::shared_expr &ex);
template void moto::ocp_base::add<moto::shared_expr>(moto::shared_expr &&ex);
template void moto::ocp_base::add<const moto::shared_expr>(const moto::shared_expr &&ex);
