#include <moto/ocp/problem.hpp>
#include <moto/solver/ineq_soft.hpp>

namespace moto {
namespace solver {
namespace ineq_soft {
void initialize(node_data *cur) {
    for_each(cur, [cur](auto &sf, soft_constr_data_t &sd) {
        sd.prim_step_.clear();
        for (const sym &arg : sf.in_args()) {
            if (arg.field() < field::num_prim) {
                sd.prim_step_.push_back(cur->problem().extract_tangent(dynamic_cast<solver::data_base *>(cur)->prim_step[arg.field()], arg));
            }
            new (&sd.d_multiplier_) mapped_vector{cur->problem().extract(
                                                      dynamic_cast<solver::data_base *>(cur)->dual_step[sf.field()], sf).data(),
                                                  sf.dim()};
        }
        sf.initialize(sd);
    });
}
void finalize_newton_step(node_data *cur, bool update_res_stat) {
    auto &prob = cur->problem();
    for_each(cur, [update_res_stat, &prob](const soft_constr &sf, soft_constr_data_t &sd) {
        sf.finalize_newton_step(sd);
    });
}
void finalize_predictor_step(node_data *data, workspace_data *config) {
    for_each(data, [cfg = config](const soft_constr &sf, soft_constr_data_t &sd) {
        sf.finalize_predictor_step(sd, cfg);
    });
}
void apply_affine_step(node_data *cur, workspace_data *config) {
    for_each(cur, [cfg = config](const soft_constr &sf, soft_constr_data_t &sd) {
        sf.apply_affine_step(sd, cfg);
    });
}
void calculate_line_search_bounds(node_data *cur, workspace_data *config) {
    for_each(cur, [cfg = config](const soft_constr &sf, soft_constr_data_t &sd) {
        sf.update_linesearch_bounds(sd, cfg);
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
    for_each(dynamic_cast<node_data *>(data), [](const soft_constr &sf, soft_constr_data_t &sd) {
        sf.apply_corrector_step(sd);
    });
    data->swap_jacobian_modification(); // move modification to the jacobian for later solving
}
void first_order_correction_end(data_base *data) {
    data->swap_jacobian_modification();                     // move
    data->Q_y_corr = &data->dense_->jac_modification_[__y]; // cache the Q_y after correction
    // data->merge_jacobian_modification();
}

} // namespace ineq_soft
} // namespace solver
} // namespace moto