#include <algorithm>
#include <atomic>
#include <optional>
#include <unordered_map>

#include <moto/ocp/problem.hpp>

namespace moto {
namespace {
size_t next_projector_group_id() {
    static std::atomic<size_t> next_id{1};
    return next_id.fetch_add(1, std::memory_order_relaxed);
}

projector_group_spec *find_projector_group(projector_layout_spec &layout,
                                           size_t group_id) {
    auto it = std::find_if(layout.groups.begin(), layout.groups.end(), [&](const projector_group_spec &group) {
        return group.id == group_id;
    });
    return it == layout.groups.end() ? nullptr : &*it;
}

const projector_group_spec *find_projector_group(const projector_layout_spec &layout,
                                                 size_t group_id) {
    auto it = std::find_if(layout.groups.begin(), layout.groups.end(), [&](const projector_group_spec &group) {
        return group.id == group_id;
    });
    return it == layout.groups.end() ? nullptr : &*it;
}

projector_group_spec &ensure_projector_group(projector_layout_spec &layout,
                                             size_t group_id) {
    if (auto *group = find_projector_group(layout, group_id)) {
        return *group;
    }
    layout.groups.push_back(projector_group_spec{.id = group_id});
    return layout.groups.back();
}

void append_unique_members(std::vector<projector_member_ref> &dst,
                           const std::vector<projector_member_ref> &src) {
    for (const auto &member : src) {
        const bool exists = std::any_of(dst.begin(), dst.end(), [&](const projector_member_ref &dst_member) {
            return dst_member.uid == member.uid && dst_member.name == member.name;
        });
        if (!exists) {
            dst.push_back(member);
        }
    }
}

void append_unique_order_edge(std::vector<projector_before_spec> &order,
                              size_t before_group_id,
                              size_t after_group_id) {
    const bool exists = std::any_of(order.begin(), order.end(), [&](const projector_before_spec &edge) {
        return edge.before == before_group_id && edge.after == after_group_id;
    });
    if (!exists) {
        order.push_back(projector_before_spec{
            .before = before_group_id,
            .after = after_group_id,
        });
    }
}

std::vector<size_t> topologically_sorted_group_indices(const projector_layout_spec &layout) {
    std::unordered_map<size_t, size_t> group_index;
    group_index.reserve(layout.groups.size());
    for (size_t i = 0; i < layout.groups.size(); ++i) {
        const auto [it, inserted] = group_index.emplace(layout.groups[i].id, i);
        if (!inserted) {
            throw std::runtime_error(fmt::format(
                "projector layout contains duplicate group id {}",
                layout.groups[i].id));
        }
    }

    std::vector<std::vector<size_t>> out_edges(layout.groups.size());
    std::vector<size_t> indegree(layout.groups.size(), 0);
    for (const auto &edge : layout.order) {
        const auto before_it = group_index.find(edge.before);
        const auto after_it = group_index.find(edge.after);
        if (before_it == group_index.end() || after_it == group_index.end()) {
            throw std::runtime_error(fmt::format(
                "projector layout references unknown group order {} -> {}",
                edge.before,
                edge.after));
        }
        if (before_it->second == after_it->second) {
            throw std::runtime_error(fmt::format(
                "projector layout has a self-order edge on group {}",
                edge.before));
        }
        auto &adj = out_edges[before_it->second];
        if (std::find(adj.begin(), adj.end(), after_it->second) == adj.end()) {
            adj.push_back(after_it->second);
            ++indegree[after_it->second];
        }
    }

    std::vector<size_t> ready;
    ready.reserve(layout.groups.size());
    for (size_t i = 0; i < indegree.size(); ++i) {
        if (indegree[i] == 0) {
            ready.push_back(i);
        }
    }

    std::vector<size_t> order;
    order.reserve(layout.groups.size());
    while (!ready.empty()) {
        const size_t next = ready.front();
        ready.erase(ready.begin());
        order.push_back(next);
        for (size_t child : out_edges[next]) {
            if (--indegree[child] == 0) {
                ready.push_back(child);
            }
        }
    }

    if (order.size() != layout.groups.size()) {
        throw std::runtime_error("projector layout has cyclic ordering requirements");
    }
    return order;
}

void reorder_exprs_by_uid(expr_list &exprs,
                          const std::vector<size_t> &ordered_uids,
                          field_t field) {
    if (exprs.size() != ordered_uids.size()) {
        throw std::runtime_error(fmt::format(
            "projector layout produced {} entries for field {} but the field stores {} expressions",
            ordered_uids.size(),
            field::name(field),
            exprs.size()));
    }

    std::unordered_map<size_t, shared_expr> by_uid;
    by_uid.reserve(exprs.size());
    for (const auto &expr : exprs) {
        by_uid.emplace(expr->uid(), expr);
    }

    expr_list ordered;
    ordered.reserve(exprs.size());
    for (size_t uid : ordered_uids) {
        const auto it = by_uid.find(uid);
        if (it == by_uid.end()) {
            throw std::runtime_error(fmt::format(
                "projector layout references uid {} missing from field {}",
                uid,
                field::name(field)));
        }
        ordered.emplace_back(it->second);
    }
    exprs.swap(ordered);
}

std::string projector_group_label(size_t group_id) {
    return fmt::format("projector_group_{}", group_id);
}

struct resolved_projector_group {
    size_t group_id = 0;
    std::string label;
    std::vector<size_t> controls;
    std::vector<size_t> projected_state_constraints;
    std::vector<size_t> state_input_constraints;
};
} // namespace

projector_layout ocp_base::projector() {
    finalized_ = false;
    return projector_layout(this);
}

void ocp_base::merge_projector_layout_from(const ocp_base &rhs) {
    for (const auto &rhs_group : rhs.projector_layout_.groups) {
        auto &dst_group = ensure_projector_group(projector_layout_, rhs_group.id);
        append_unique_members(dst_group.primals, rhs_group.primals);
        append_unique_members(dst_group.constraints, rhs_group.constraints);
    }
    for (const auto &edge : rhs.projector_layout_.order) {
        append_unique_order_edge(projector_layout_.order, edge.before, edge.after);
    }
    finalized_ = false;
}

void ocp_base::compile_projector_layout() {
    const auto control_exprs = expr_[__u];
    const auto projected_state_exprs = expr_[__eq_x];
    const auto state_input_exprs = expr_[__eq_xu];

    std::unordered_map<size_t, size_t> control_authored_index;
    std::unordered_map<size_t, size_t> projected_state_authored_index;
    std::unordered_map<size_t, size_t> state_input_authored_index;
    std::unordered_map<std::string, size_t> control_uid_by_name;
    std::unordered_map<std::string, size_t> projected_state_uid_by_name;
    std::unordered_map<std::string, size_t> state_input_uid_by_name;
    control_authored_index.reserve(control_exprs.size());
    projected_state_authored_index.reserve(projected_state_exprs.size());
    state_input_authored_index.reserve(state_input_exprs.size());
    control_uid_by_name.reserve(control_exprs.size());
    projected_state_uid_by_name.reserve(projected_state_exprs.size());
    state_input_uid_by_name.reserve(state_input_exprs.size());

    auto record_unique_name = [](auto &by_name, const shared_expr &expr) {
        const auto [it, inserted] = by_name.emplace(expr->name(), expr->uid());
        if (!inserted && it->second != expr->uid()) {
            it->second = static_cast<size_t>(-1);
        }
    };

    for (size_t i = 0; i < control_exprs.size(); ++i) {
        control_authored_index.emplace(control_exprs[i]->uid(), i);
        record_unique_name(control_uid_by_name, control_exprs[i]);
    }
    for (size_t i = 0; i < projected_state_exprs.size(); ++i) {
        projected_state_authored_index.emplace(projected_state_exprs[i]->uid(), i);
        record_unique_name(projected_state_uid_by_name, projected_state_exprs[i]);
    }
    for (size_t i = 0; i < state_input_exprs.size(); ++i) {
        state_input_authored_index.emplace(state_input_exprs[i]->uid(), i);
        record_unique_name(state_input_uid_by_name, state_input_exprs[i]);
    }

    std::unordered_map<size_t, size_t> control_owner;
    std::unordered_map<size_t, size_t> constraint_owner;
    control_owner.reserve(control_exprs.size());
    constraint_owner.reserve(projected_state_exprs.size() + state_input_exprs.size());

    std::vector<resolved_projector_group> resolved_groups;

    auto claim_uid = [&](auto &owners,
                         size_t uid,
                         const std::string &kind,
                         size_t owner_group_id) {
        const auto [it, inserted] = owners.emplace(uid, owner_group_id);
        if (!inserted && it->second != owner_group_id) {
            throw std::runtime_error(fmt::format(
                "projector layout conflict: {} uid {} is owned by both group {} and group {}",
                kind,
                uid,
                it->second,
                owner_group_id));
        }
    };

    auto resolve_member_uid = [](const projector_member_ref &member,
                                 const auto &uid_index,
                                 const auto &name_index) -> std::optional<size_t> {
        if (uid_index.contains(member.uid)) {
            return member.uid;
        }
        const auto name_it = name_index.find(member.name);
        if (name_it != name_index.end() && name_it->second != static_cast<size_t>(-1)) {
            return name_it->second;
        }
        return std::nullopt;
    };

    const auto group_order = topologically_sorted_group_indices(projector_layout_);
    for (size_t group_idx : group_order) {
        const auto &group = projector_layout_.groups[group_idx];
        resolved_projector_group resolved{
            .group_id = group.id,
            .label = projector_group_label(group.id),
        };

        for (const auto &member : group.primals) {
            const auto resolved_uid = resolve_member_uid(member, control_authored_index, control_uid_by_name);
            if (!resolved_uid.has_value()) {
                throw std::runtime_error(fmt::format(
                    "projector group {} references control {}:{} that is not present in the realized stage",
                    group.id,
                    member.name,
                    member.uid));
            }
            claim_uid(control_owner, *resolved_uid, "control", group.id);
            resolved.controls.push_back(*resolved_uid);
        }
        std::sort(resolved.controls.begin(), resolved.controls.end(), [&](size_t lhs, size_t rhs) {
            return control_authored_index.at(lhs) < control_authored_index.at(rhs);
        });
        resolved.controls.erase(std::unique(resolved.controls.begin(), resolved.controls.end()), resolved.controls.end());

        for (const auto &member : group.constraints) {
            const auto state_uid = resolve_member_uid(member,
                                                      projected_state_authored_index,
                                                      projected_state_uid_by_name);
            const auto input_uid = resolve_member_uid(member,
                                                      state_input_authored_index,
                                                      state_input_uid_by_name);
            const auto state_it = state_uid.has_value()
                                      ? projected_state_authored_index.find(*state_uid)
                                      : projected_state_authored_index.end();
            const auto input_it = input_uid.has_value()
                                      ? state_input_authored_index.find(*input_uid)
                                      : state_input_authored_index.end();
            if (state_it == projected_state_authored_index.end() &&
                input_it == state_input_authored_index.end()) {
                std::string available_constraints;
                bool first = true;
                for (const auto &expr : projected_state_exprs) {
                    fmt::format_to(std::back_inserter(available_constraints),
                                   "{}{}:{}@{}",
                                   first ? "" : ", ",
                                   expr->name(),
                                   expr->uid(),
                                   field::name(__eq_x));
                    first = false;
                }
                for (const auto &expr : state_input_exprs) {
                    fmt::format_to(std::back_inserter(available_constraints),
                                   "{}{}:{}@{}",
                                   first ? "" : ", ",
                                   expr->name(),
                                   expr->uid(),
                                   field::name(__eq_xu));
                    first = false;
                }
                throw std::runtime_error(fmt::format(
                    "projector group {} references hard-constraint uid {} that is not present in the realized stage; available hard constraints are [{}]",
                    group.id,
                    member.uid,
                    available_constraints));
            }
            const size_t resolved_uid = state_it != projected_state_authored_index.end()
                                            ? *state_uid
                                            : *input_uid;
            claim_uid(constraint_owner, resolved_uid, "hard constraint", group.id);
            if (state_it != projected_state_authored_index.end()) {
                resolved.projected_state_constraints.push_back(*state_uid);
            } else {
                resolved.state_input_constraints.push_back(*input_uid);
            }
        }
        std::sort(resolved.projected_state_constraints.begin(),
                  resolved.projected_state_constraints.end(),
                  [&](size_t lhs, size_t rhs) {
                      return projected_state_authored_index.at(lhs) < projected_state_authored_index.at(rhs);
                  });
        resolved.projected_state_constraints.erase(
            std::unique(resolved.projected_state_constraints.begin(),
                        resolved.projected_state_constraints.end()),
            resolved.projected_state_constraints.end());
        std::sort(resolved.state_input_constraints.begin(),
                  resolved.state_input_constraints.end(),
                  [&](size_t lhs, size_t rhs) {
                      return state_input_authored_index.at(lhs) < state_input_authored_index.at(rhs);
                  });
        resolved.state_input_constraints.erase(
            std::unique(resolved.state_input_constraints.begin(),
                        resolved.state_input_constraints.end()),
            resolved.state_input_constraints.end());

        if (!resolved.controls.empty() ||
            !resolved.projected_state_constraints.empty() ||
            !resolved.state_input_constraints.empty()) {
            resolved_groups.push_back(std::move(resolved));
        }
    }

    if (!resolved_groups.empty()) {
        std::vector<size_t> ordered_controls;
        ordered_controls.reserve(control_exprs.size());
        for (const auto &group : resolved_groups) {
            ordered_controls.insert(ordered_controls.end(), group.controls.begin(), group.controls.end());
        }
        for (const auto &expr : control_exprs) {
            if (!control_owner.contains(expr->uid())) {
                ordered_controls.push_back(expr->uid());
            }
        }
        if (!ordered_controls.empty()) {
            reorder_exprs_by_uid(expr_[__u], ordered_controls, __u);
        }
    }

    const bool has_constraint_groups = std::any_of(resolved_groups.begin(), resolved_groups.end(), [](const resolved_projector_group &group) {
        return !group.projected_state_constraints.empty() || !group.state_input_constraints.empty();
    });
    if (!has_constraint_groups) {
        return;
    }

    std::vector<size_t> ordered_projected_state;
    ordered_projected_state.reserve(projected_state_exprs.size());
    std::vector<size_t> ordered_state_input;
    ordered_state_input.reserve(state_input_exprs.size());

    size_t projected_state_offset = 0;
    size_t state_input_offset = 0;
    for (const auto &group : resolved_groups) {
        if (!group.projected_state_constraints.empty()) {
            ordered_projected_state.insert(ordered_projected_state.end(),
                                           group.projected_state_constraints.begin(),
                                           group.projected_state_constraints.end());
            compiled_projector_layout_.hard_constraint_blocks.push_back(projector_hard_constraint_block{
                .field = __eq_x,
                .source_begin = projected_state_offset,
                .source_count = group.projected_state_constraints.size(),
                .group_id = group.group_id,
                .group = group.label,
            });
            projected_state_offset += group.projected_state_constraints.size();
        }
        if (!group.state_input_constraints.empty()) {
            ordered_state_input.insert(ordered_state_input.end(),
                                       group.state_input_constraints.begin(),
                                       group.state_input_constraints.end());
            compiled_projector_layout_.hard_constraint_blocks.push_back(projector_hard_constraint_block{
                .field = __eq_xu,
                .source_begin = state_input_offset,
                .source_count = group.state_input_constraints.size(),
                .group_id = group.group_id,
                .group = group.label,
            });
            state_input_offset += group.state_input_constraints.size();
        }
    }

    for (const auto &expr : projected_state_exprs) {
        if (!constraint_owner.contains(expr->uid())) {
            ordered_projected_state.push_back(expr->uid());
        }
    }
    for (const auto &expr : state_input_exprs) {
        if (!constraint_owner.contains(expr->uid())) {
            ordered_state_input.push_back(expr->uid());
        }
    }

    if (!ordered_projected_state.empty()) {
        reorder_exprs_by_uid(expr_[__eq_x], ordered_projected_state, __eq_x);
    }
    if (!ordered_state_input.empty()) {
        reorder_exprs_by_uid(expr_[__eq_xu], ordered_state_input, __eq_xu);
    }

    const size_t projected_state_tail = ordered_projected_state.size() - projected_state_offset;
    const size_t state_input_tail = ordered_state_input.size() - state_input_offset;
    if (projected_state_tail > 0) {
        compiled_projector_layout_.hard_constraint_blocks.push_back(projector_hard_constraint_block{
            .field = __eq_x,
            .source_begin = projected_state_offset,
            .source_count = projected_state_tail,
            .group_id = 0,
            .group = "__eq_x_projected",
        });
    }
    if (state_input_tail > 0) {
        compiled_projector_layout_.hard_constraint_blocks.push_back(projector_hard_constraint_block{
            .field = __eq_xu,
            .source_begin = state_input_offset,
            .source_count = state_input_tail,
            .group_id = 0,
            .group = "__eq_xu",
        });
    }
}

projector_group projector_layout::group() {
    if (owner_ == nullptr) {
        throw std::runtime_error("projector_layout is not attached to a problem");
    }
    const size_t group_id = next_projector_group_id();
    ensure_projector_group(owner_->projector_layout_, group_id);
    owner_->finalized_ = false;
    return projector_group(owner_, group_id);
}

projector_group &projector_group::require_primal(const var_inarg_list &vars) {
    if (owner_ == nullptr || id_ == 0) {
        throw std::runtime_error("projector_group is not attached to a problem");
    }
    auto &group = ensure_projector_group(owner_->projector_layout_, id_);
    for (const sym &var : vars) {
        group.primals.push_back(projector_member_ref{
            .uid = var.uid(),
            .name = var.name(),
        });
    }
    owner_->finalized_ = false;
    return *this;
}

projector_group &projector_group::require_constraint(const expr_inarg_list &exprs) {
    if (owner_ == nullptr || id_ == 0) {
        throw std::runtime_error("projector_group is not attached to a problem");
    }
    auto &group = ensure_projector_group(owner_->projector_layout_, id_);
    for (const expr &entry : exprs) {
        group.constraints.push_back(projector_member_ref{
            .uid = entry.uid(),
            .name = entry.name(),
        });
    }
    owner_->finalized_ = false;
    return *this;
}

projector_group &projector_group::require_before(const projector_group &after) {
    if (owner_ == nullptr || id_ == 0 || after.id_ == 0) {
        throw std::runtime_error("projector_group ordering requires two attached groups");
    }
    append_unique_order_edge(owner_->projector_layout_.order, id_, after.id_);
    owner_->finalized_ = false;
    return *this;
}

projector_group &projector_group::require_after(const projector_group &before) {
    if (owner_ == nullptr || id_ == 0 || before.id_ == 0) {
        throw std::runtime_error("projector_group ordering requires two attached groups");
    }
    append_unique_order_edge(owner_->projector_layout_.order, before.id_, id_);
    owner_->finalized_ = false;
    return *this;
}
} // namespace moto
