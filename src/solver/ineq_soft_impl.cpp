#include <moto/solver/ineq_soft.hpp>
#include <moto/ocp/problem.hpp>

namespace moto {
namespace solver {
namespace ineq_soft {
void initialize(node_data *cur) {
    for_each(cur, [cur](auto &sf, auto &sd) {
        sd.prim_step_.clear();
        for (const sym &arg : sf.in_args()) {
            if (arg.field() < field::num_prim) {
                sd.prim_step_.push_back(cur->problem().extract(dynamic_cast<solver::data_base*>(cur)->prim_step[arg.field()], arg));
            }
        }
        sf.initialize(sd);
    });
}
void finalize_newton_step(node_data *cur) {
    for_each(cur, [](auto &sf, auto &sd) {
        sf.finalize_newton_step(sd);
    });
}
void finalize_predictor_step(node_data *data, workspace_data *config) {
    for_each(data, [cfg = config](const soft_constr &sf, soft_constr_data_t &sd) {
        sf.finalize_predictor_step(sd, cfg);
    });
}
void line_search_step(node_data *cur, workspace_data *config) {
    for_each(cur, [cfg = config](const soft_constr &sf, soft_constr_data_t &sd) {
        sf.line_search_step(sd, cfg);
    });
}
void calculate_line_search_bounds(node_data *cur, workspace_data *config) {
    for_each(cur, [cfg = config](const soft_constr &sf, soft_constr_data_t &sd) {
        sf.update_linesearch_config(sd, cfg);
    });
}
void first_order_correction_start(data_base *data) {
    data->Q_y_corr = nullptr;
    data->prim_corr[__x].setZero();
    // data->clear_merit_jac();
    // clear modification
    for (auto field : primal_fields) {
        data->dense_->jac_modification_[field].setZero();
    }
    for_each(dynamic_cast<node_data*>(data), [](const soft_constr &sf, soft_constr_data_t &sd) {
        sf.correct_jacobian(sd);
    });
    // data->merge_jacobian_modification();
    data->swap_jacobian_modification(); // move modification to the jacobian for later solving
}
void first_order_correction_end(data_base *data) {
    data->swap_jacobian_modification();                     // move
    data->Q_y_corr = &data->dense_->jac_modification_[__y]; // cache the Q_y after correction
}
} // namespace ineq_soft
} // namespace solver
} // namespace moto