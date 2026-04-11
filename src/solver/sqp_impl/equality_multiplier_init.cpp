#include <moto/solver/equality_init/eq_init_overlay.hpp>
#include <moto/solver/ineq_soft.hpp>
#include <moto/solver/ns_sqp.hpp>

namespace moto {
namespace {

/*
Equality-multiplier initialization is implemented as a one-shot solve on a
restoration-style overlay graph.

Design summary:
- reuse the realized outer x/u/y chain
- keep __dyn hard
- replace hard __eq_x / __eq_xu with PMM overlay constraints
- keep existing __eq_x_soft / __eq_xu_soft as cloned soft constraints
- keep inequalities active in the overlay and sync their IPM state from outer
- warm-start overlay duals from the current outer iterate
- run one accepted Newton step with line search disabled and restoration disabled
- optionally run equality-init-specific iterative refinement
- copy back only equality-type multipliers

The outer problem must keep fixed:
- primal x/u/y
- inequality multipliers
- inequality slack / bound state
*/

template <typename Fn>
void for_each_overlay_pair(ns_sqp::storage_type &outer_graph,
                           ns_sqp::storage_type &overlay_graph,
                           Fn &&fn) {
    auto &outer_nodes = outer_graph.flatten_nodes();
    auto &overlay_nodes = overlay_graph.flatten_nodes();
    if (outer_nodes.size() != overlay_nodes.size()) {
        throw std::runtime_error("equality-init overlay graph/node mismatch");
    }
    for (size_t i = 0; i < outer_nodes.size(); ++i) {
        fn(*outer_nodes[i], *overlay_nodes[i]);
    }
}

bool graph_has_equality_targets(ns_sqp::storage_type &graph) {
    for (auto *node : graph.flatten_nodes()) {
        for (auto field : std::array{__dyn, __eq_x, __eq_xu, __eq_x_soft, __eq_xu_soft}) {
            if (node->problem().dim(field) > 0) {
                return true;
            }
        }
    }
    return false;
}

struct scoped_eq_init_settings {
    ns_sqp::settings_t &settings;
    struct backup_t {
        bool ls_enabled;
        bool restoration_enabled;
        ns_sqp::iterative_refinement_setting rf;
    } backup;

    explicit scoped_eq_init_settings(ns_sqp::settings_t &s)
        : settings(s), backup{s.ls.enabled, s.restoration.enabled, s.rf} {
        settings.ls.enabled = false;
        settings.restoration.enabled = false;
        settings.rf = settings.eq_init.rf;
    }

    ~scoped_eq_init_settings() {
        settings.ls.enabled = backup.ls_enabled;
        settings.restoration.enabled = backup.restoration_enabled;
        settings.rf = backup.rf;
    }
};

} // namespace

void ns_sqp::initialize_equality_multipliers() {
    if (!settings.eq_init.enabled) {
        return;
    }

    auto &outer_graph = active_data();
    if (!graph_has_equality_targets(outer_graph)) {
        return;
    }

    auto &overlay_graph = equality_init_graph();
    scoped_eq_init_settings scoped_settings(settings);
    const bool was_in_restoration = settings.in_restoration;
    settings.in_restoration = false;
    set_phase_graph_override(overlay_graph);
    try {
        for_each_overlay_pair(outer_graph, overlay_graph, [&](data &outer, data &overlay) {
            solver::equality_init::sync_equality_init_overlay_primal(outer, overlay);
        });

        overlay_graph.for_each_parallel([this](data *d) {
            d->for_each_constr([this](const generic_func &c, func_approx_data &fd) { c.setup_workspace_data(fd, &settings); });
            solver::ineq_soft::bind_and_invalidate(d);
            d->update_approximation(node_data::update_mode::eval_val, true);
        });

        for_each_overlay_pair(outer_graph, overlay_graph, [&](data &outer, data &overlay) {
            solver::equality_init::sync_equality_init_overlay_duals(outer, overlay);
        });

        overlay_graph.for_each_parallel([](data *d) {
            d->update_approximation(node_data::update_mode::eval_all, true);
        });

        kkt_info kkt_overlay;
        update_primal_info(kkt_overlay, point_value_mask::primal);
        filter_linesearch_data ls;
        ls.constr_vio_min = std::max(kkt_overlay.primal.res_l1 * settings.ls.constr_vio_min_frac, settings.prim_tol);
        sqp_iter(ls, kkt_overlay, /*do_scaling=*/false, /*do_refinement=*/settings.rf.enabled);

        for_each_overlay_pair(outer_graph, overlay_graph, [&](data &outer, data &overlay) {
            solver::equality_init::commit_equality_init_overlay_duals(outer, overlay);
        });
    } catch (...) {
        settings.in_restoration = was_in_restoration;
        clear_phase_graph_override();
        throw;
    }
    settings.in_restoration = was_in_restoration;
    clear_phase_graph_override();

    outer_graph.for_each_parallel([](data *d) {
        d->update_approximation(node_data::update_mode::eval_derivatives, true);
    });
}

} // namespace moto
