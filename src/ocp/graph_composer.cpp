#include <moto/ocp/graph_composer.hpp>

#include <algorithm>
#include <moto/ocp/impl/func.hpp>

namespace moto {
void graph_composer::append_placement(
    const stage_ocp_ptr_t &source, stage_expr_role role,
    term_placement placement, const ocp_ptr_t &target,
    status_request_list &status) const {
    if (!source)
        return;
    source->wait_until_ready();
    const auto request = [&](expr_handle expression, bool active) {
        const auto found = std::find_if(
            status.begin(), status.end(), [&](const status_request &request) {
                return request.expression->uid() == expression->uid();
            });
        if (found == status.end())
            status.push_back({std::move(expression), active});
        else
            found->active = found->active || active;
    };

    for (field_t f : func_fields) {
        for (const expr_handle &expr : source->exprs(f)) {
            if (!source->has_role(*expr, role))
                continue;
            const auto function = expr_cast<generic_func>(expr);
            expr_handle placed = expr;
            if (placement == term_placement::lower_x_to_y) {
                placed = function->lower_expr_x_to_y_reuse(
                    fmt::format("stage endpoint term {} materialization", expr->name()),
                    target->uid());
            }
            target->add(placed);
            request(placed, true);

            for (const sym &arg : function->in_args()) {
                if (!in_field(arg.field(), primal_fields)) {
                    continue;
                }
                expr_handle mapped = arg.handle();
                if (placement == term_placement::lower_x_to_y) {
                    if (arg.field() != __x) {
                        throw std::logic_error(
                            "endpoint lowering encountered a non-x primal argument");
                    }
                    mapped = arg.next()->handle();
                }
                request(std::move(mapped), source->is_active(arg));
            }
        }
    }
}

void graph_composer::resolve_status(const status_request_list &status,
                                    const ocp_ptr_t &target) const {
    ocp::active_status_config config;
    expr_list resolved_functions;
    config.activate_list.reserve(status.size());
    config.deactivate_list.reserve(status.size());
    resolved_functions.reserve(status.size());
    for (const auto &request : status) {
        if (request.active != target->is_active(*request.expression)) {
            auto &list = request.active ? config.activate_list : config.deactivate_list;
            list.emplace_back(request.expression);
        }
        if (in_field(request.expression->field(), func_fields)) {
            resolved_functions.emplace_back(request.expression);
        }
    }
    target->resolve_composed_status(config, resolved_functions);
}

size_t graph_composer::composition_key_hash::operator()(const composition_key &key) const noexcept {
    size_t hash = 0;
    for (const auto &identity : key.identity) {
        const size_t value = std::hash<const void *>{}(identity.get());
        hash ^= value + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
    }
    return hash;
}

ocp_ptr_t graph_composer::compose_stage(
    const graph_model::topology_snapshot &topology, size_t index) const {
    const auto &record = (*topology.stages)[index];
    const size_t count = topology.stages->size();
    auto composed = ocp::create();
    composed->set_allow_inconsistent_dynamics(record.stage->allow_inconsistent_dynamics());
    composed->set_automatic_reorder_primal(record.stage->automatic_reorder_primal());

    status_request_list status;
    append_placement(record.stage, stage_expr_role::interval,
                     term_placement::direct, composed, status);
    if (index == 0)
        append_placement(topology.start.stage, stage_expr_role::start_node,
                         term_placement::direct, composed, status);
    append_placement(record.stage, stage_expr_role::end_node,
                     term_placement::lower_x_to_y, composed, status);
    append_placement(index + 1 < count ? (*topology.stages)[index + 1].stage
                                      : topology.end.stage,
                     index + 1 < count ? stage_expr_role::start_node
                                       : stage_expr_role::end_node,
                     term_placement::lower_x_to_y, composed, status);
    resolve_status(status, composed);
    composed->wait_until_ready();
    return composed;
}

graph_composer::interval_snapshot graph_composer::compose(const graph_model &model) const {
    const auto topology = model.snapshot();
    std::lock_guard lock(mutex_);
    if (cache_revision_ == topology.revision && interval_cache_) {
        return {cache_revision_, interval_cache_};
    }
    auto intervals = std::make_shared<std::vector<composed_interval>>();
    std::unordered_map<composition_key, ocp_ptr_t, composition_key_hash> next;
    intervals->reserve(topology.stages->size());
    next.reserve(entries_.size() + 4);
    for (size_t i = 0; i < topology.stages->size(); ++i) {
        composition_key cache_key{{
            (*topology.stages)[i].identity,
            i == 0 ? topology.start.identity : nullptr,
            i + 1 < topology.stages->size() ? (*topology.stages)[i + 1].identity : nullptr,
            i + 1 == topology.stages->size() ? topology.end.identity : nullptr,
        }};
        ocp_ptr_t composed;
        if (const auto found = next.find(cache_key); found != next.end()) {
            composed = found->second;
        } else if (const auto found = entries_.find(cache_key); found != entries_.end()) {
            composed = found->second;
        } else {
            composed = compose_stage(topology, i);
        }
        next.emplace(std::move(cache_key), composed);
        intervals->push_back({(*topology.stages)[i].stage, std::move(composed)});
    }
    entries_ = std::move(next);
    interval_cache_ = intervals;
    cache_revision_ = topology.revision;
    return {cache_revision_, interval_cache_};
}

} // namespace moto
