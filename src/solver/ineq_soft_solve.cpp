#include <moto/ocp/constr.hpp>
#include <moto/solver/ineq_soft_solve.hpp>

namespace moto {
namespace ineq_soft_solve {

template <typename Callback>
    requires std::is_invocable_v<Callback, soft_constr_impl &, soft_constr_data &>
static inline void for_funcs(solver::data_base *data, Callback &&callback) {
    auto inner = [&](field_t f) {
        for (auto &func : data->ocp_->expr_[f]) {
            soft_constr_impl &sf = static_cast<soft_constr_impl &>(*func);
            auto &sd = static_cast<soft_constr_data &>(data->data(sf));
            callback(sf, sd);
        }
    };
    for (auto f : solver::ineq_constr_fields)
        inner(f);
    for (auto f : solver::soft_constr_fields)
        inner(f);
}

void initialize(solver::data_base *cur) {
    for_funcs(cur, [cur](auto &sf, auto &sd) {
        sf.initialize(sd);
    });
}
void post_rollout(solver::data_base *cur) {
    for_funcs(cur, [cur](auto &sf, auto &sd) {
        sf.post_rollout(sd);
    });
}
void line_search_step(solver::data_base *cur, scalar_t alpha) {
    for_funcs(cur, [cur, alpha](soft_constr_impl &sf, soft_constr_data &sd) {
        sf.line_search_step(sd, alpha);
    });
}
 
} // namespace ineq_soft_solve
} // namespace moto