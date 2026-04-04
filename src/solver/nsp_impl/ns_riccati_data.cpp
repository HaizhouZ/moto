#include <moto/ocp/impl/node_data.hpp>
#include <moto/solver/ns_riccati/ns_riccati_data.hpp>
#include <moto/solver/ns_riccati/generic_solver.hpp>
#include <moto/ocp/dynamics.hpp>

namespace moto {
namespace solver {
namespace ns_riccati {
void ns_riccati_data::update_projected_dynamics() {
    full_data_->for_each(__dyn, [](const generic_dynamics &dyn, func_approx_data &data) {
        dyn.compute_project_jacobians(data);
    });
}
void ns_riccati_data::update_projected_dynamics_residual() {
    full_data_->for_each(__dyn, [](const generic_dynamics &dyn, func_approx_data &data) {
        dyn.compute_project_residual(data);
    });
}
void ns_riccati_data::apply_jac_y_inverse_transpose(vector &v, vector &dst) {
    full_data_->for_each(__dyn, [&v, &dst](const generic_dynamics &dyn, func_approx_data &data) {
        dyn.apply_jac_y_inverse_transpose(data, v, dst);
    });
}

ns_riccati_data generic_solver::create_data(node_data *full_data) {
    return ns_riccati_data(full_data);
}

ns_riccati_data::ns_riccati_data(node_data *full_data)
    : solver::data_base(&full_data->sym_val(), &full_data->dense()),
      full_data_(full_data),
      ns(dense_->approx_[__eq_x].v_.size()),
      nc(dense_->approx_[__eq_xu].v_.size()), ncstr(ns + nc), d_u(nu, nx),
      d_y(nx, nx), d_lbd_f(nx), d_lbd_s_c_pre_solve(nu), d_lbd_s_c(ncstr),
      default_elimination_stage_(dense_->approx_[__eq_x].v_.size(),
                                 dense_->approx_[__eq_xu].v_.size(),
                                 nu),
      F_x(dense_->proj_f_x()),
      F_u(dense_->proj_f_u()),
      s_y(dense_->approx_[__eq_x].jac_[__y]),
      s_x(dense_->approx_[__eq_x].jac_[__x]),
      c_x(dense_->approx_[__eq_xu].jac_[__x]),
      c_u(dense_->approx_[__eq_xu].jac_[__u]),
      F_0(dense_->proj_f_res()) {
    if (nu < ncstr) {
        nz = 0;
        // throw std::runtime_error("system over-constrained, i.e., nu < ncstr");
    } else {
        nz = nu - ncstr;
    }
    // nsp_->F_0_k.resize(nx);
    // nsp_->F_0_K.resize(nx, nx);
    // nsp_->F_u.resize(nx, nu);
    nsp_.Q_zz.resize(nu, nu);
    nsp_.s_0_p_k.resize(ns);
    nsp_.s_0_p_K.resize(ns, nx);
    nsp_.y_y_k.resize(nx);
    nsp_.y_y_K.resize(nx, nx);
    nsp_.u_y_k.resize(nu);
    nsp_.u_y_K.resize(nu, nx);
    nsp_.u_0_p_k.resize(nu);
    nsp_.u_0_p_K.resize(nu, nx);
    nsp_.y_0_p_k.resize(nx);
    nsp_.y_0_p_K.resize(nx, nx);
    nsp_.s_u.resize(ns, nu);
    nsp_.s_c_stacked.resize(ncstr, nu);
    nsp_.s_c_stacked_0_k.resize(ncstr);
    nsp_.s_c_stacked_0_K.resize(ncstr, nx);
    d_y.K.setZero();
    const auto &hard_constraint_blocks = full_data_->problem().compiled_hard_constraint_blocks();
    if (!hard_constraint_blocks.empty()) {
        std::vector<projection::equality_block_spec> stage_blocks;
        stage_blocks.reserve(hard_constraint_blocks.size());
        for (const auto &block : hard_constraint_blocks) {
            const auto kind = block.field == __eq_x
                                  ? projection::equality_block_kind::projected_state
                                  : projection::equality_block_kind::state_input;
            stage_blocks.push_back(projection::equality_block_spec{
                .kind = kind,
                .source_begin = block.source_begin,
                .source_count = block.source_count,
                .group = block.group,
            });
        }
        default_elimination_stage_.configure_blocks(ns, nc, nu, stage_blocks);
    }
    // if (nsp_->sparse_factorizer_)
    // nsp_->sparse_factorizer_->init(nsp_);
}

} // namespace ns_riccati
} // namespace solver
} // namespace moto
