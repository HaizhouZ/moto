#include <algorithm>
#include <array>
#include <moto/ocp/dynamics.hpp>
#include <moto/ocp/problem.hpp>

namespace moto {
INIT_UID_(ocp_base);

namespace {
constexpr auto status_func_fields = concat_fields(func_fields, custom_func_fields);

bool has_active_primal_arg(const generic_func &func, const ocp_base *prob) {
    for (const sym &arg : func.in_args()) {
        if (in_field(arg.field(), primal_fields) && prob->is_active(arg)) {
            return true;
        }
    }
    return false;
}

bool can_be_active_by_status(const generic_func &func, ocp_base *prob) {
    return has_active_primal_arg(func, prob) && func.check_enable(prob);
}

const generic_func *as_generic_func(const shared_expr &ex) {
    return ex ? dynamic_cast<const generic_func *>(ex.get()) : nullptr;
}

bool is_pure_y_func(const generic_func &func) {
    bool has_y = false;
    for (const sym &arg : func.in_args()) {
        if (!in_field(arg.field(), primal_fields)) {
            continue;
        }
        if (arg.field() != __y) {
            return false;
        }
        has_y = true;
    }
    return has_y;
}

constexpr unsigned role_bit(stage_expr_role role) {
    return 1u << static_cast<unsigned>(role);
}

} // namespace

ocp_base::ocp_base() { uid_.set_inc(); }

ocp_base::ocp_base(const ocp_base &rhs)
    : field_layout_store<expr_list>(rhs),
      finalized_(rhs.finalized_),
      uid_(rhs.uid_),
      disabled_expr_(rhs.disabled_expr_),
      pruned_expr_(rhs.pruned_expr_),
      uids_(rhs.uids_),
      disabled_uids_(rhs.disabled_uids_),
      pruned_uids_(rhs.pruned_uids_),
      allow_inconsistent_dynamics_(rhs.allow_inconsistent_dynamics_),
      automatic_reorder_primal_(rhs.automatic_reorder_primal_) {}

ocp_base::~ocp_base() = default;

bool ocp_base::add_impl(expr &ex) {
    return add_impl(shared_expr(ex));
}
bool ocp_base::add_impl(shared_expr ex) {
    if (!ex) {
        throw std::runtime_error("Cannot add null expression to problem");
    }
    std::string reason;
    if (!accepts_term(ex, &reason)) {
        if (reason.empty()) {
            reason = "expression is incompatible with this problem type";
        }
        throw std::runtime_error(fmt::format(
            "Cannot add expression {} uid {} to problem uid {}: {}",
            ex->name(), ex->uid(), uid_, reason));
    }
    size_t _uid = ex->uid();
    if (!contains(*ex)) {
        if (!ex->finalize()) {
            throw std::runtime_error(fmt::format("cannot finalize expr {} uid {}", ex->name(), ex->uid()));
        }
        const auto &dep = ex->dep();
        if (!dep.empty()) {
            if (ex->field() == __dyn && !allow_inconsistent_dynamics_)
                for (auto f : {__x, __y})
                    for (const generic_dynamics &dyn : field_entries(__dyn)) {
                        auto it = std::ranges::find_if(dep, [&](const sym &s) { return dyn.has_arg(s); });
                        if (it != dep.end()) {
                            throw std::runtime_error(
                                fmt::format("Dynamics {} arg {} uid {} in {} found in dynamics {}. "
                                            "Overlapping state variables in dynamics is not allowed to avoid inconsistency."
                                            " If you want to allow this, call set_allow_inconsistent_dynamics(true).",
                                            ex->name(),
                                            (*it)->name(), (*it)->uid(), f, dyn.name()));
                        }
                    }
            for (expr &arg : dep) {
                if (!contains(arg)) {
                    add_impl(arg);
                }
            }
        }
        finalized_ = false;
        if (ex->default_active_status()) {
            uids_.insert(_uid);
            append_entry(std::move(ex));
        } else {
            disabled_uids_.insert(_uid);
            disabled_expr_[ex->field()].emplace_back(std::move(ex));
        }
        on_modified();
        return true;
    }
    return false;
}
bool ocp_base::contains(const expr &ex) const {
    return uids_.contains(ex.uid()) ||
           disabled_uids_.contains(ex.uid()) ||
           pruned_uids_.contains(ex.uid());
}
bool ocp_base::is_active(const expr &ex) const { return uids_.contains(ex.uid()); }
const expr_list &ocp_base::exprs(size_t f) const { return field_entries(f); }
size_t ocp_base::pos(const expr &ex) const {
    field_read_guard();
    return entry_index(ex);
}
size_t ocp_base::dim(size_t f) const { field_read_guard(); return field_dim(f); }
size_t ocp_base::num(size_t f) const { return field_entry_count(f); }
size_t ocp_base::tdim(size_t f) const { field_read_guard(); return field_tdim(f); }
size_t ocp_base::get_expr_start(const expr &ex) const {
    field_read_guard();
    return field_start(ex);
}
size_t ocp_base::get_expr_start_tangent(const expr &ex) const {
    field_read_guard();
    return field_tangent_start(ex);
}
void ocp_base::finalize() {
    static std::mutex finalize_mutex_;
    std::lock_guard lock(finalize_mutex_);
    if (!finalized_) {
        if (!field_empty(__dyn) && automatic_reorder_primal_)
            maintain_order();
        rebuild_layout();
        this->finalized_ = true;
    }
}
void ocp_base::refresh_after_clone(const active_status_config &config) {
    finalized_ = false;
    if (!config.empty()) {
        update_active_status(config);
    }
}
void ocp_base::on_modified() {}
void ocp_base::maintain_order() {
    expr_list tmp;
    for (auto f : {__x, __y}) {
        auto &syms = field_entries(f);
        tmp.reserve(syms.size());
        for (const generic_func &dyn : exprs(__dyn)) {
            for (const expr &arg : dyn.in_args(f)) {
                auto it = std::find(syms.begin(), syms.end(), arg);
                if (it == syms.end()) {
                    throw std::runtime_error(fmt::format(
                        "order maintenance failure: "
                        "Dynamics {} arg {} uid {} not found in field {}",
                        dyn.name(), arg.name(), arg.uid(), f));
                }
                tmp.emplace_back(std::move(*it));
            }
        }
        std::erase_if(syms, [&](auto &&e) { return !e; });
        if (!syms.empty()) {
            throw std::runtime_error(fmt::format(
                "order maintenance failure: "
                " field {} has exprs not in dynamics args",
                f));
        }
        syms.swap(tmp);
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
    for (const auto &f : field_entries()) {
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
bool ocp_base::accepts_term(const shared_expr &ex, std::string *reason) const {
    static_cast<void>(ex);
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

stage_ocp::stage_ocp(const stage_ocp &rhs)
    : ocp(rhs),
      endpoint_role_mask_by_uid_(rhs.endpoint_role_mask_by_uid_) {}

stage_ocp_ptr_t stage_ocp::clone(const active_status_config &config) const {
    auto prob = stage_ocp_ptr_t(new stage_ocp(*this));
    prob->refresh_after_clone(config);
    return prob;
}
bool stage_ocp::add_with_role(shared_expr ex, stage_expr_role role) {
    if (!ex) {
        throw std::runtime_error("Cannot add null expression to stage_ocp");
    }
    const size_t uid = ex->uid();
    const auto existing_it = endpoint_role_mask_by_uid_.find(uid);
    const bool had_existing = existing_it != endpoint_role_mask_by_uid_.end();
    const unsigned existing = had_existing ? existing_it->second : 0u;
    if (role == stage_expr_role::interval) {
        if (had_existing) {
            throw std::runtime_error(fmt::format(
                "Cannot add expression {} uid {} to both interval and endpoint stage roles",
                ex->name(), uid));
        }
        return ocp_base::add_impl(std::move(ex));
    }
    if (!had_existing && contains(*ex)) {
        throw std::runtime_error(fmt::format(
            "Cannot add expression {} uid {} to both interval and endpoint stage roles",
            ex->name(), uid));
    }
    const unsigned updated = existing | role_bit(role);
    endpoint_role_mask_by_uid_[uid] = updated;
    try {
        const bool added = ocp_base::add_impl(std::move(ex));
        if (!added && updated != existing) {
            on_modified();
        }
        return added;
    } catch (...) {
        if (!had_existing) {
            endpoint_role_mask_by_uid_.erase(uid);
        } else {
            endpoint_role_mask_by_uid_[uid] = existing;
        }
        throw;
    }
}
bool stage_ocp::validate_stage_term(const shared_expr &ex, std::string *reason) const {
    if (reason != nullptr) {
        reason->clear();
    }
    if (!ex) {
        if (reason != nullptr) *reason = "null expression";
        return false;
    }
    const auto *func = as_generic_func(ex);
    if (func == nullptr) {
        return true;
    }
    if (is_pure_y_func(*func)) {
        if (reason != nullptr) {
            *reason = "pure y-only terms should be written on x and added through stage.ed.add(...)";
        }
        return false;
    }
    return true;
}
bool stage_ocp::validate_endpoint_term(const shared_expr &ex, std::string *reason) const {
    if (reason != nullptr) {
        reason->clear();
    }
    if (!ex) {
        if (reason != nullptr) *reason = "null expression";
        return false;
    }
    const auto *func = as_generic_func(ex);
    if (func == nullptr || ex->field() == __dyn || dynamic_cast<const generic_dynamics *>(ex.get()) != nullptr) {
        if (reason != nullptr) {
            *reason = "node_view only accepts pure x cost/constraint terms; dynamics belong to stage.add(...)";
        }
        return false;
    }
    if (!func->has_pure_x_primal_args()) {
        if (reason != nullptr) {
            *reason = "node_view only accepts terms with x/prm-style dependencies and no u or y arguments";
        }
        return false;
    }
    return true;
}
bool stage_ocp::accepts_term(const shared_expr &ex, std::string *reason) const {
    return validate_stage_term(ex, reason);
}
bool stage_ocp::has_role(const expr &ex, stage_expr_role role) const {
    if (auto it = endpoint_role_mask_by_uid_.find(ex.uid()); it != endpoint_role_mask_by_uid_.end()) {
        return (it->second & role_bit(role)) != 0u;
    }
    return role == stage_expr_role::interval;
}
void stage_ocp::set_mutation_callback(std::function<void()> callback) {
    mutation_callback_ = std::move(callback);
}
void stage_ocp::on_modified() {
    if (mutation_callback_) {
        mutation_callback_();
    }
}
node_view stage_ocp::st() {
    return node_view(shared_from_this(), stage_expr_role::start_node);
}
node_view stage_ocp::ed() {
    return node_view(shared_from_this(), stage_expr_role::end_node);
}

node_view::node_view(const stage_ocp_ptr_t &stage, stage_expr_role role)
    : owner_(stage), role_(role) {}

void ocp_base::move_active_expr(const expr &ex, bool prune) {
    const size_t f = ex.field();
    auto &target_store = prune ? pruned_expr_ : disabled_expr_;
    auto &target_uids = prune ? pruned_uids_ : disabled_uids_;
    target_store[f].emplace_back(take_field_entry(ex));
    target_uids.insert(ex.uid());
    uids_.erase(ex.uid());
}
bool ocp_base::restore_inactive_expr(const expr &ex, bool from_pruned) {
    const size_t f = ex.field();
    auto &source_store = from_pruned ? pruned_expr_ : disabled_expr_;
    auto &source_uids = from_pruned ? pruned_uids_ : disabled_uids_;
    auto &source_list = source_store[f];
    if (find_entry(source_list, ex) == source_list.end()) {
        return false;
    }
    append_entry(take_entry(source_list, ex));
    uids_.insert(ex.uid());
    source_uids.erase(ex.uid());
    return true;
}
void ocp_base::update_active_status(const active_status_config &config) {
    for (expr &ex : config.activate_list) {
        if (!restore_inactive_expr(ex, true) && !restore_inactive_expr(ex, false)) {
            throw std::runtime_error(fmt::format("Cannot activate expression {} uid {}, it does not exist in the problem",
                                                 ex.name(), ex.uid()));
        }
    }
    for (const expr &ex : config.deactivate_list) {
        move_active_expr(ex, false);
    }
    for (int remaining = 5;; --remaining) {
        if (remaining == 0) {
            throw std::runtime_error("ocp::clone failed to converge during pruning");
        }
        bool changed = false;
        array_type<std::vector<std::reference_wrapper<const expr>>, status_func_fields> to_delete, to_re_enable;
        for (auto f : status_func_fields) {
            if (pruned_expr_[f].empty())
                continue;
            to_re_enable[f].reserve(pruned_expr_[f].size());
            for (const generic_func &e : pruned_expr_[f]) {
                if (can_be_active_by_status(e, this)) {
                    to_re_enable[f].emplace_back(e);
                }
            }
        }
        for (auto f : status_func_fields) {
            if (field_empty(f))
                continue;
            to_delete[f].reserve(field_entry_count(f));
            for (const generic_func &e : field_entries(f)) {
                if (!can_be_active_by_status(e, this)) {
                    to_delete[f].emplace_back(e);
                }
            }
        }
        for (auto f : status_func_fields) {
            for (const expr &e : to_delete[f]) {
                move_active_expr(e, true);
                if (std::find(config.activate_list.begin(), config.activate_list.end(), e) != config.activate_list.end()) {
                    throw std::runtime_error(fmt::format("func {} uid {} pruned but also in activate_list",
                                                         e.name(), e.uid()));
                }
                changed = true;
            }
            for (const expr &e : to_re_enable[f]) {
                restore_inactive_expr(e, true);
                if (std::find(config.deactivate_list.begin(), config.deactivate_list.end(), e) != config.deactivate_list.end()) {
                    throw std::runtime_error(fmt::format("func {} uid {} re-enabled but also in deactivate_list",
                                                         e.name(), e.uid()));
                }
                changed = true;
            }
        }
        if (!changed) {
            break;
        }
    }
    finalized_ = false;
    on_modified();
}
} // namespace moto

template void moto::ocp_base::add<const moto::shared_expr &>(const moto::shared_expr &ex);
template void moto::ocp_base::add<moto::shared_expr>(moto::shared_expr &&ex);
template void moto::ocp_base::add<const moto::shared_expr>(const moto::shared_expr &&ex);
