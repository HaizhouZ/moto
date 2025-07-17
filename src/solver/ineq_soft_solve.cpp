#include <moto/ocp/impl/soft_constr.hpp>
#include <moto/solver/ineq_soft_solve.hpp>

namespace moto {
namespace ineq_soft_solve {
using soft_constr = impl::soft_constr;
using soft_constr_data_t = soft_constr::sp_data_map;
template <typename Callback>
    requires std::is_invocable_v<Callback, soft_constr &, soft_constr_data_t &>
static inline void for_funcs(solver::data_base *data, Callback &&callback) {
    auto inner = [&](field_t f) {
        for (auto &func : data->prob_->expr_[f]) {
            soft_constr &sf = static_cast<soft_constr &>(*func);
            auto &sd = dynamic_cast<soft_constr_data_t &>(data->data(sf));
            callback(sf, sd);
        }
    };
    for (auto f : concat_fields(ineq_constr_fields, soft_constr_fields))
        inner(f);
}

void initialize(solver::data_base *cur) {
    for_funcs(cur, [cur](auto &sf, auto &sd) {
        sf.initialize(sd);
    });
}
void finalize_newton_step(solver::data_base *cur) {
    for_funcs(cur, [cur](auto &sf, auto &sd) {
        sf.finalize_newton_step(sd);
    });
}
void line_search_step(solver::data_base *cur, workspace_data *config) {
    for_funcs(cur, [cur, cfg = config](soft_constr &sf, soft_constr_data_t &sd) {
        sf.line_search_step(sd, cfg);
    });
}
void calculate_line_search_bounds(solver::data_base *cur, workspace_data *config) {
    for_funcs(cur, [cur, cfg = config](soft_constr &sf, soft_constr_data_t &sd) {
        sf.update_linesearch_config(sd, cfg);
    });
}
} // namespace ineq_soft_solve
} // namespace moto