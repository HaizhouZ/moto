#include <moto/solver/ipm.hpp>
#include <moto/solver/ipm_constr.hpp>

namespace moto {
namespace ipm {
void post_rollout(solver::data_base *cur) {
    // get all ipm constraints
    for (auto &p : cur->sparse_[__ineq_x]) {
        auto &d = static_cast<ipm_data &>(*p);
        size_t arg_idx = 0;
        // update slack newton step
        d.d_slack_ = d.v_ + d.slack_; // +r_g
        // compute linear step
        for (const auto &arg : d.func_.in_args()) {
            if (arg->field_ < field::num_prim) {
                d.d_slack_.noalias() -= d.jac_[arg_idx] * cur->ocp_->extract(cur->prim_step[arg->field_], *arg);
            }
            arg_idx++;
        }
        // update dual newton step
        d.d_multipler_.array() = - d.comp_res_.array() - d.diag_scaling.array() * d.d_slack_.array();
    }
}

} // namespace ipm
} // namespace moto
