#include <moto/solver/ns_sqp.hpp>
#define stat_width 15

namespace moto {

struct stat_item {
    std::string_view name;
    size_t width; // default width for each stat item
    stat_item(std::string_view n, size_t w = stat_width) : name(n), width(w) {}
    void print_header() const {
        fmt::print("| {:<{}} |", name, width);
    }
};
stat_item stats[] = {{"no.", 3},
                     {"objective"},
                     {"prim_res"},
                     {"dual_res"},
                     {"comp_res"},
                     {"||p||"},
                     {"||d||"},
                     {"alpha_p"},
                     {"alpha_d"},
                     {"ls", 3},
                     {"ipm_mu", stat_width + 2}};

void ns_sqp::print_stat_header() {
    for (const auto &term : stats) {
        term.print_header();
    }
    putc('\n', stdout);
}

void ns_sqp::print_stats(int i_iter, const kkt_info &info, bool hcast_ineq) {
    scalar_t stats_value[] = {0., info.objective, info.inf_prim_res, info.inf_dual_res, info.inf_comp_res, info.inf_prim_step, info.inf_dual_step,
                              settings.alpha_primal, settings.alpha_dual, 0., settings.mu};
    std::string_view ipm_flags;
    if (hcast_ineq && settings.ipm_enable_corrector()) {
        if (settings.ipm_accept_corrector()) {
            ipm_flags = "[c:a]";
        } else {
            ipm_flags = "[c:r]";
        }
    }
    size_t idx_stat = 0;
    for (auto &item : stats) {
        if (item.name == "no.") {
            fmt::print("| {:<{}} |", i_iter < 0 ? "--" : std::to_string(i_iter + 1), item.width);
        } else if (item.name == "ls") {
            fmt::print("| {:<{}} |", info.ls_steps < 0 ? "--" : std::to_string(info.ls_steps), item.width);
        } else if (item.name == "ipm_mu") {
            fmt::print("| {:<{}} |", hcast_ineq ? fmt::format("{:.6e}{}", stats_value[idx_stat], ipm_flags) : "---------", item.width);
        } else {
            fmt::print("| {:<{}.6e} |", stats_value[idx_stat], item.width);
        }
        idx_stat++;
    }

    fmt::print("\n");
};
} // namespace moto