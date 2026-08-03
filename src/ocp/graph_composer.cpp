#include <moto/ocp/graph_composer.hpp>

#include <moto/ocp/impl/func.hpp>

namespace moto {
namespace {
void deactivate_from(const stage_ocp_ptr_t &source, const ocp_ptr_t &target) {
    if (!source)
        return;
    ocp::active_status_config config;
    for (field_t f : primal_fields) {
        for (const expr_handle &expr : target->exprs(f)) {
            if (source->contains(*expr) && !source->is_active(*expr)) {
                config.deactivate_list.emplace_back(expr);
            }
        }
    }
    if (!config.empty())
        target->update_active_status(config);
}
} // namespace

void graph_composer::append_role_terms(const stage_ocp_ptr_t &source, stage_expr_role role, const ocp_ptr_t &target,
                                       term_placement placement) const {
    if (!source)
        return;
    for (field_t f : func_fields) {
        for (const expr_handle &expr : source->exprs(f)) {
            if (!source->has_role(*expr, role))
                continue;
            if (placement == term_placement::direct) {
                target->add(expr);
            } else {
                target->add(expr_cast<generic_func>(expr)->lower_expr_x_to_y_reuse(
                    fmt::format("stage endpoint term {} materialization", expr->name()), target->uid()));
            }
        }
    }
}

void graph_composer::append_node_terms(const node_view &node, const ocp_ptr_t &target, term_placement placement) const {
    if (const auto source = node.stage()) {
        source->wait_until_ready();
        append_role_terms(source, node.role(), target, placement);
    }
}

std::vector<graph_composer::dependency> graph_composer::dependencies(const graph_model::interval_record &record) const {
    std::vector<dependency> result;
    result.reserve(1 + bool(record.start_boundary_view) + record.end_boundary_views.size());
    auto append = [&](const stage_ocp_ptr_t &stage, stage_expr_role role) {
        if (stage)
            result.push_back({stage.get(), stage->mutation_revision(), role});
    };
    append(record.stage, stage_expr_role::interval);
    append(record.start_boundary_view.stage(), record.start_boundary_view.role());
    for (const node_view &view : record.end_boundary_views)
        append(view.stage(), view.role());
    return result;
}

ocp_ptr_t graph_composer::compose_stage(const graph_model::interval_record &record) const {
    if (!record.stage)
        throw std::runtime_error("graph composer found null stage");
    record.stage->wait_until_ready();
    auto composed = ocp::create();
    composed->set_allow_inconsistent_dynamics(record.stage->allow_inconsistent_dynamics());
    composed->set_automatic_reorder_primal(record.stage->automatic_reorder_primal());
    append_role_terms(record.stage, stage_expr_role::interval, composed, term_placement::direct);
    if (record.start_boundary_view) {
        append_node_terms(record.start_boundary_view, composed, term_placement::direct);
    }
    deactivate_from(record.stage, composed);
    if (record.start_boundary_view)
        deactivate_from(record.start_boundary_view.stage(), composed);
    for (const node_view &view : record.end_boundary_views) {
        append_node_terms(view, composed, term_placement::lower_x_to_y);
    }
    composed->wait_until_ready();
    return composed;
}

graph_composer::interval_snapshot graph_composer::compose(const graph_model &model) const {
    for (;;) {
        const auto topology = model.snapshot();
        std::lock_guard lock(mutex_);
        if (cache_revision_ == topology.revision && interval_cache_) {
            return {cache_revision_, interval_cache_};
        }
        auto intervals = std::make_shared<std::vector<ocp_ptr_t>>();
        std::vector<cache_entry> next;
        intervals->reserve(topology.intervals->size());
        next.reserve(topology.intervals->size());
        for (size_t i = 0; i < topology.intervals->size(); ++i) {
            const auto deps = dependencies((*topology.intervals)[i]);
            if (i < entries_.size() && entries_[i].composed && entries_[i].dependencies == deps) {
                next.push_back(entries_[i]);
            } else {
                next.push_back({deps, compose_stage((*topology.intervals)[i])});
            }
            intervals->push_back(next.back().composed);
        }
        if (model.revision() == topology.revision) {
            entries_ = std::move(next);
            interval_cache_ = intervals;
            cache_revision_ = topology.revision;
            return {cache_revision_, interval_cache_};
        }
    }
}

} // namespace moto
