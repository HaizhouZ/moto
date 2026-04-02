#include <moto/solver/ns_sqp.hpp>
#define stat_width 15

namespace moto {

namespace {
struct phase_mu_info {
    scalar_t display = 0.;
    scalar_t min = 0.;
    scalar_t max = 0.;
    bool valid = false;
};

phase_mu_info current_phase_mu(ns_sqp::solver_graph_type &graph,
                               bool in_restoration,
                               scalar_t outer_mu) {
    phase_mu_info info;
    if (!in_restoration) {
        info.display = outer_mu;
        info.min = info.display;
        info.max = info.display;
        info.valid = true;
        return info;
    }

    for (auto *n : graph.flatten_nodes()) {
        const auto *aux =
            dynamic_cast<const ns_sqp::ns_riccati_data::restoration_aux_data *>(n->aux_.get());
        if (aux == nullptr) {
            continue;
        }
        if (!info.valid) {
            info.min = aux->mu_bar;
            info.max = aux->mu_bar;
            info.valid = true;
        } else {
            info.min = std::min(info.min, aux->mu_bar);
            info.max = std::max(info.max, aux->mu_bar);
        }
    }
    if (info.valid) {
        // Restoration owns one mu_bar per stage. The stats table has one scalar column,
        // so we report the conservative aggregate max(mu_bar) and print the range below.
        info.display = info.max;
    } else {
        info.display = outer_mu;
        info.min = outer_mu;
        info.max = outer_mu;
        info.valid = true;
    }
    return info;
}
} // namespace

struct stat_item {
    std::string_view name;
    size_t width;
    int precision; // -1 means use default (.6e)
    stat_item(std::string_view n, size_t w = stat_width, int p = -1) : name(n), width(w), precision(p) {}
    void print_header() const {
        fmt::print("| {:<{}} |", name, width);
    }
};
stat_item stats[] = {{"no.", 3},
                     {"obj", 8, 3},
                     {"r(prim)", 8, 3},
                     {"r(dual)", 8, 3},
                     {"r(comp)", 8, 3},
                     {"||p||", 8, 3},
                     {"||d_eq||", 8, 3},
                     {"||d_iq||", 8, 3},
                     {"alpha_p", 8, 3},
                     {"alpha_d", 8, 3},
                     {"ls", 3},
                     {"mu(max)", stat_width + 2}};

void ns_sqp::print_stat_header() {
    for (const auto &term : stats) {
        term.print_header();
    }
    putc('\n', stdout);
}

void ns_sqp::print_stats(const kkt_info &info) {
    auto &graph = solver_graph();
    const bool restoration_active = in_restoration_phase();
    auto mu_info = current_phase_mu(graph, restoration_active, settings.ipm.mu);
    if (!restoration_active && !settings.has_ineq_soft) {
        mu_info.valid = false;
    }
    scalar_t stats_value[] = {0., info.penalized_obj, info.inf_prim_res, info.inf_dual_res, info.inf_comp_res, info.inf_prim_step, info.inf_eq_dual_step, info.inf_ineq_dual_step,
                              settings.ls.alpha_primal, settings.ls.alpha_dual, 0., mu_info.display};
    std::string_view ipm_flags;
    if (!restoration_active && settings.has_ineq_soft && settings.ipm.ipm_enable_corrector()) {
        if (settings.ipm.ipm_accept_corrector()) {
            ipm_flags = "[c:a]";
        } else {
            ipm_flags = "[c:r]";
        }
    }
    size_t idx_stat = 0;
    size_t i_iter = static_cast<size_t>(info.num_iter);
    for (auto &item : stats) {
        if (item.name == "no.") {
            fmt::print("| {:<{}} |", i_iter == 0 ? "--" : std::to_string(i_iter), item.width);
        } else if (item.name == "ls") {
            fmt::print("| {:<{}} |", info.ls_steps < 0 ? "--" : std::to_string(info.ls_steps), item.width);
        } else if (item.name == "mu(max)") {
            if (!mu_info.valid) {
                fmt::print("| {:<{}} |", "---------", item.width);
            } else if (restoration_active) {
                fmt::print("| {:<{}} |", fmt::format("{:.3e}(resto)", stats_value[idx_stat]), item.width);
            } else {
                fmt::print("| {:<{}} |", fmt::format("{:.3e}{}({:.1f})", stats_value[idx_stat], ipm_flags, std::log10(stats_value[idx_stat])), item.width);
            }
        } else if (item.precision == 3) {
            fmt::print("| {:<{}.3e} |", stats_value[idx_stat], item.width);
        } else {
            fmt::print("| {:<{}.6e} |", stats_value[idx_stat], item.width);
        }
        idx_stat++;
    }

    fmt::print("\n");
    if (restoration_active && mu_info.valid && std::abs(mu_info.max - mu_info.min) > scalar_t(1e-15)) {
        fmt::print("    restoration mu_bar range=[{:.3e}, {:.3e}] (table shows max)\n",
                   mu_info.min, mu_info.max);
    }
    fmt::print("    ||lam_eq||={:.3e}  ||lam_ineq||={:.3e}  diag_scl={:.3e}  ||lam||={:.3e}\n",
               info.max_eq_dual_norm, info.max_ineq_dual_norm, info.max_diag_scaling, info.max_dual_norm);
    fmt::print("    d_eq: dyn={:.3e}  eq_x={:.3e}  eq_xu={:.3e}\n",
               info.inf_dyn_dual_step, info.inf_eq_x_dual_step, info.inf_eq_xu_dual_step);
    std::vector<vector> dual_dyn;
    size_t idx = 0;
    if (i_iter == 0)
        return;
    for (auto n : graph.flatten_nodes()) {
        // fmt::print("    value x at node {}: {:.4}\n", idx, n->trial_prim_state_bak[__x].transpose());
        // fmt::print("    delta x at node {}: {:.4}\n", idx, n->trial_prim_step[__x].transpose());
        // fmt::print("    value u at node {}: {:.4}\n", idx, n->trial_prim_state_bak[__u].transpose());
        // fmt::print("    delta u at node {}: {:.4}\n", idx, n->trial_prim_step[__u].transpose());
        // fmt::print("    value y at node {}: {:.4}\n", idx, n->trial_prim_state_bak[__y].transpose());
        // fmt::print("    delta y at node {}: {:.4}\n", idx, n->trial_prim_step[__y].transpose());

        // if (n->dense().dual_[__dyn].size() > 0) {
        //     dual_dyn.push_back(n->trial_dual_step[__dyn]);
        //     fmt::print("    lam_dyn at node {}: {:.4}\n", idx, n->trial_dual_state_bak[__dyn].transpose());
        //     fmt::print("    d l_dyn at node {}: {:.4}\n", idx, n->trial_dual_step[__dyn].transpose());
        // }
        // // if (idx == 99) {
        //     fmt::print("    lam_eqc at node {}: {:.4}\n", idx + 1, n->trial_dual_state_bak[__eq_x_soft].transpose());
        //     fmt::print("    d l_eqc at node {}: {:.4}\n", idx + 1, n->trial_dual_step[__eq_x_soft].transpose());
        // // }
        // fmt::println("    node {}:", idx);
        // fmt::print("R \n{}\n", n->Q_uu.dense());
        // fmt::print("q \n{}\n", n->Q_u.transpose());
        // fmt::print("Q \n{}\n", n->Q_yy.dense());
        // fmt::print("r \n{}\n", n->Q_y.transpose());
        idx++;
    }
};
} // namespace moto
