#include <moto/ocp/problem.hpp>
#include <moto/solver/ineq_soft.hpp>

namespace moto {
namespace solver {
namespace ineq_soft {
void initialize(node_data *cur) {
    for_each(cur, [cur](auto &&sf, auto &&sd) {
        sd.prim_step_.clear();
        auto prob = sd.problem();
        auto d = dynamic_cast<solver::data_base *>(cur);
        assert(d && "data_base cast failed");
        for (const sym &arg : sf.in_args()) {
            if (arg.field() < field::num_prim && prob->is_active(arg)) {
                sd.prim_step_.push_back(prob->extract_tangent(d->trial_prim_step[arg.field()], arg));
            } else {
                static vector empty;
                sd.prim_step_.emplace_back(empty);
            }
        }
        new (&sd.d_multiplier_) mapped_vector{
            prob->extract(d->trial_dual_step[sf.field()], sf).data(), Eigen::Index(sf.dim())};
        sf.initialize(sd);
    });
}

void finalize_newton_step(node_data *cur) {
    for_each(cur, [](auto &&sf, auto &&sd) {
        sf.finalize_newton_step(sd);
    });
}
void finalize_predictor_step(node_data *data, workspace_data *config) {
    for_each(data, [config](auto &&sf, auto &&sd) {
        sf.finalize_predictor_step(sd, config);
    });
}
void apply_affine_step(node_data *cur, workspace_data *config) {
    for_each(cur, [config](auto &&sf, auto &&sd) {
        sf.apply_affine_step(sd, config);
    });
}
void update_ls_bounds(node_data *cur, workspace_data *config) {
    for_each(cur, [config](auto &&sf, auto &&sd) {
        sf.update_ls_bounds(sd, config);
    });
}
void backup_trial_state(node_data *cur) {
    for_each(cur, [](auto &&sf, auto &&sd) {
        sf.backup_trial_state(sd);
    });
}
void restore_trial_state(node_data *cur) {
    for_each(cur, [](auto &&sf, auto &&sd) {
        sf.restore_trial_state(sd);
    });
}
void corrector_step_start(data_base *data) {
    data->first_order_correction_start(
        get_for_each([](auto &&sf, auto &&sd) {
            sf.apply_corrector_step(sd);
        }));
}
void corrector_step_end(data_base *data) {
    data->first_order_correction_end();
}

} // namespace ineq_soft
} // namespace solver
} // namespace moto