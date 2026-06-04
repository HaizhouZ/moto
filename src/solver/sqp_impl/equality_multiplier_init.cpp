#include <moto/solver/equality_init/eq_init_overlay.hpp>
#include <moto/solver/ineq_soft.hpp>
#include <moto/solver/ns_sqp.hpp>

namespace moto {
namespace {

bool graph_has_equality_targets(ns_sqp::storage_type &graph) {
    for (node_data *node : graph.flatten_nodes()) {
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

bool ns_sqp::initialize_equality_multipliers(storage_type &outer_graph, bool refresh_outer_derivatives) {
    if (!settings.eq_init.enabled) {
        return false;
    }

    if (!graph_has_equality_targets(outer_graph)) {
        return false;
    }

    auto &overlay_graph = equality_init_graph();
    scoped_eq_init_settings scoped_settings(settings);
    scoped_phase_graph_override phase_graph(*this, overlay_graph, false);
    solver::for_each(solver::par, solver::zip(outer_graph, overlay_graph),
                     [&](data *outer, data *overlay) {
                         solver::equality_init::sync_equality_init_overlay_primal(*outer, *overlay);
                         overlay->for_each_constr([this](const generic_constr &c, func_approx_data &fd) { c.setup_workspace_data(fd, &settings); });
                         solver::ineq_soft::bind_and_invalidate(overlay);
                         solver::equality_init::sync_equality_init_overlay_duals(*outer, *overlay);
                         solver::ineq_soft::mark_initialized(overlay);
                         overlay->update_approximation(node_data::update_mode::eval_all, true);
                     });

    kkt_info kkt_overlay;
    update_primal_info(kkt_overlay, point_value_mask::primal);
    filter_linesearch_data ls;
    ls.constr_vio_min = std::max(kkt_overlay.primal.res_l1 * settings.ls.constr_vio_min_frac, settings.prim_tol);
    sqp_iter(ls, kkt_overlay, /*do_scaling=*/false, /*do_refinement=*/settings.rf.enabled);

    solver::for_each(solver::par, solver::zip(outer_graph, overlay_graph),
                     [refresh_outer_derivatives](data *outer, data *overlay) {
                         solver::equality_init::commit_equality_init_overlay_duals(*outer, *overlay);
                         if (refresh_outer_derivatives) {
                             outer->update_approximation(node_data::update_mode::eval_derivatives, true);
                         }
                     });
    return true;
}

} // namespace moto
