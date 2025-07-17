#include <moto/ocp/impl/soft_constr.hpp>
#include <moto/solver/ineq_soft_solve.hpp>

namespace moto {
namespace ineq_soft_solve {
using soft_constr = impl::soft_constr;
using soft_constr_data_t = soft_constr::sp_data_map;
void initialize(solver::data_base *cur) {
    for_each(cur, [](auto &sf, auto &sd) {
        sf.initialize(sd);
    });
}
void finalize_newton_step(solver::data_base *cur) {
    for_each(cur, [](auto &sf, auto &sd) {
        sf.finalize_newton_step(sd);
    });
}
void line_search_step(solver::data_base *cur, workspace_data *config) {
    for_each(cur, [cfg = config](soft_constr &sf, soft_constr_data_t &sd) {
        sf.line_search_step(sd, cfg);
    });
}
void calculate_line_search_bounds(solver::data_base *cur, workspace_data *config) {
    for_each(cur, [cfg = config](soft_constr &sf, soft_constr_data_t &sd) {
        sf.update_linesearch_config(sd, cfg);
    });
}
void correct_jacobian(solver::data_base *data) {
    for_each(data, [](soft_constr &sf, soft_constr_data_t &sd) {
        sf.correct_jacobian(sd);
    });
}
} // namespace ineq_soft_solve
} // namespace moto