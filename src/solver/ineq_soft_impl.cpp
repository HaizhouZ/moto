#include <moto/ocp/impl/soft_constr.hpp>
#include <moto/solver/ineq_soft.hpp>

namespace moto {
namespace solver {
namespace ineq_soft {
void initialize(data_base *cur) {
    for_each(cur, [cur](auto &sf, auto &sd) {
        sd.prim_step_.clear();
        for (const auto &arg : sf.in_args()) {
            if (arg->field_ < field::num_prim) {
                sd.prim_step_.push_back(cur->prob_->extract(cur->prim_step[arg->field_], *arg));
            }
        }
        sf.initialize(sd);
    });
}
void finalize_newton_step(data_base *cur) {
    for_each(cur, [](auto &sf, auto &sd) {
        sf.finalize_newton_step(sd);
    });
}
void line_search_step(data_base *cur, workspace_data *config) {
    for_each(cur, [cfg = config](soft_constr &sf, soft_constr_data_t &sd) {
        sf.line_search_step(sd, cfg);
    });
}
void calculate_line_search_bounds(data_base *cur, workspace_data *config) {
    for_each(cur, [cfg = config](soft_constr &sf, soft_constr_data_t &sd) {
        sf.update_linesearch_config(sd, cfg);
    });
}
void first_order_correction(data_base *data) {
    data->Q_y_cache = data->Q_y; // cache the Q_y before correction
    data->prim_corr[__x].setZero();
    data->clear_merit_jac();
    for_each(data, [](soft_constr &sf, soft_constr_data_t &sd) {
        sf.correct_jacobian(sd);
    });
    data->merge_jacobian_modification();
}
} // namespace ineq_soft
} // namespace solver
} // namespace moto