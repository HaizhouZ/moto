#include <moto/solver/ns_sqp.hpp>
#define stat_width 15

namespace moto {

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
                     {"ipm_mu", stat_width + 2}};

void ns_sqp::print_stat_header() {
    for (const auto &term : stats) {
        term.print_header();
    }
    putc('\n', stdout);
}

void ns_sqp::print_stats(const kkt_info &info) {
    scalar_t stats_value[] = {0., info.cost, info.inf_prim_res, info.inf_dual_res, info.inf_comp_res, info.inf_prim_step, info.inf_eq_dual_step, info.inf_ineq_dual_step,
                              settings.ls.alpha_primal, settings.ls.alpha_dual, 0., settings.ipm.mu};
    std::string_view ipm_flags;
    if (settings.has_ineq_soft && settings.ipm.ipm_enable_corrector()) {
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
            if (settings.in_restoration) {
                fmt::print("| {:<{}} |", fmt::format("{}r", i_iter), item.width);
            } else {
                fmt::print("| {:<{}} |", i_iter == 0 ? "--" : std::to_string(i_iter), item.width);
            }
        } else if (item.name == "ls") {
            fmt::print("| {:<{}} |", info.ls_steps < 0 ? "--" : std::to_string(info.ls_steps), item.width);
        } else if (item.name == "ipm_mu") {
            fmt::print("| {:<{}} |", settings.has_ineq_soft ? fmt::format("{:.3e}{}({:.1f})", stats_value[idx_stat], ipm_flags, std::log10(stats_value[idx_stat])) : "---------", item.width);
        } else if (item.precision == 3) {
            fmt::print("| {:<{}.3e} |", stats_value[idx_stat], item.width);
        } else {
            fmt::print("| {:<{}.6e} |", stats_value[idx_stat], item.width);
        }
        idx_stat++;
    }

    fmt::print("\n");
    fmt::print("    ||lam_eq||={:.3e}  ||lam_ineq||={:.3e}  diag_scl={:.3e}  ||lam||={:.3e}\n",
               info.max_eq_dual_norm, info.max_ineq_dual_norm, info.max_diag_scaling, info.max_dual_norm);
};
} // namespace moto