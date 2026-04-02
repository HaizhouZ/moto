#include <moto/ocp/impl/node_data.hpp>
#include <moto/solver/ipm/ipm_constr.hpp>
#include <moto/solver/restoration/resto_local_kkt.hpp>
#include <moto/solver/restoration/resto_runtime.hpp>
#include <moto/spmm/sparse_mat.hpp>

namespace moto::solver::restoration {

namespace {
auto *get_aux(ns_riccati::ns_riccati_data &d) {
    return dynamic_cast<ns_riccati::ns_riccati_data::restoration_aux_data *>(d.aux_.get());
}

const auto *get_aux(const ns_riccati::ns_riccati_data &d) {
    return dynamic_cast<const ns_riccati::ns_riccati_data::restoration_aux_data *>(d.aux_.get());
}

vector gather_field_dual(const ns_riccati::ns_riccati_data &d, field_t first, field_t second) {
    const auto n_first = static_cast<Eigen::Index>(d.dense_->dual_[first].size());
    const auto n_second = static_cast<Eigen::Index>(d.dense_->dual_[second].size());
    vector out(n_first + n_second);
    if (n_first > 0) {
        out.head(n_first) = d.dense_->dual_[first];
    }
    if (n_second > 0) {
        out.tail(n_second) = d.dense_->dual_[second];
    }
    return out;
}

void scatter_field_dual(ns_riccati::ns_riccati_data &d,
                        field_t first,
                        field_t second,
                        const vector_const_ref &value,
                        bool step) {
    const auto n_first = static_cast<Eigen::Index>(d.dense_->dual_[first].size());
    const auto n_second = static_cast<Eigen::Index>(d.dense_->dual_[second].size());
    if (value.size() != n_first + n_second) {
        throw std::runtime_error("scatter_field_dual size mismatch");
    }
    auto &dst_first = step ? d.trial_dual_step[first] : d.dense_->dual_[first];
    auto &dst_second = step ? d.trial_dual_step[second] : d.dense_->dual_[second];
    if (n_first > 0) {
        dst_first = value.head(n_first);
    } else if (step) {
        dst_first.resize(0);
    }
    if (n_second > 0) {
        dst_second = value.tail(n_second);
    } else if (step) {
        dst_second.resize(0);
    }
}

vector gather_raw_ineq_values(const ns_riccati::ns_riccati_data &d) {
    vector out(static_cast<Eigen::Index>(d.dense_->approx_[__ineq_x].v_.size() +
                                         d.dense_->approx_[__ineq_xu].v_.size()));
    Eigen::Index offset = 0;
    d.full_data_->for_each<ineq_constr_fields>(
        [&](const soft_constr &, soft_constr::data_map_t &sd) {
            if (const auto *ipm = dynamic_cast<const solver::ipm_constr::approx_data *>(&sd); ipm != nullptr) {
                out.segment(offset, ipm->g_.size()) = ipm->g_;
                offset += ipm->g_.size();
            } else {
                out.segment(offset, sd.v_.size()) = sd.v_;
                offset += sd.v_.size();
            }
        });
    if (offset != out.size()) {
        out.conservativeResize(offset);
    }
    return out;
}

void build_ineq_stack(ns_riccati::ns_riccati_data &d, matrix &G_u, matrix &G_x) {
    const Eigen::Index nis = static_cast<Eigen::Index>(d.dense_->approx_[__ineq_x].v_.size());
    const Eigen::Index nic = static_cast<Eigen::Index>(d.dense_->approx_[__ineq_xu].v_.size());
    const Eigen::Index n_total = nis + nic;
    G_u.resize(n_total, d.nu);
    G_x.resize(n_total, d.nx);
    G_u.setZero();
    G_x.setZero();
    if (nis > 0) {
        matrix g_u_y(nis, d.nu);
        g_u_y.setZero();
        d.dense_->approx_[__ineq_x].jac_[__y].times<false>(d.F_u, g_u_y);
        G_u.topRows(nis) = g_u_y;
        d.dense_->approx_[__ineq_x].jac_[__x].dump_into(G_x.topRows(nis), spmm::dump_config{.overwrite = true});
        matrix g_x_y(nis, d.nx);
        g_x_y.setZero();
        d.dense_->approx_[__ineq_x].jac_[__y].times<false>(d.F_x, g_x_y);
        G_x.topRows(nis) += g_x_y;
    }
    if (nic > 0) {
        d.dense_->approx_[__ineq_xu].jac_[__u].dump_into(G_u.bottomRows(nic), spmm::dump_config{.overwrite = true});
        d.dense_->approx_[__ineq_xu].jac_[__x].dump_into(G_x.bottomRows(nic), spmm::dump_config{.overwrite = true});
    }
}

void add_diag_quadratic(row_vector &dst,
                        const vector &diag,
                        const vector &delta,
                        scalar_t scale) {
    if (dst.size() == 0 || diag.size() == 0) {
        return;
    }
    dst.noalias() += (scale * diag.array() * delta.array()).matrix().transpose();
}

void add_diag_hessian(aligned_vector_map_t &diag_block,
                      const vector &diag,
                      scalar_t scale) {
    if (diag.size() == 0) {
        return;
    }
    if (diag_block.size() != diag.size()) {
        throw std::runtime_error("add_diag_hessian diagonal size mismatch");
    }
    diag_block.get().array() += scale * diag.array();
}

void add_diag_hessian(sparse_mat &block,
                      const vector &diag,
                      scalar_t scale) {
    if (diag.size() == 0) {
        return;
    }
    const auto dim = static_cast<size_t>(diag.size());
    if (block.rows() == 0 && block.cols() == 0) {
        block.resize(dim, dim);
    }
    if (block.rows() != dim || block.cols() != dim) {
        throw std::runtime_error("add_diag_hessian dimension mismatch");
    }
    if (block.diag_panels_.empty()) {
        block.insert(0, 0, dim, dim, sparsity::diag);
    }
    if (block.diag_panels_.size() != 1 ||
        block.diag_panels_.front().row_st_ != 0 ||
        block.diag_panels_.front().col_st_ != 0 ||
        static_cast<Eigen::Index>(block.diag_panels_.front().rows_) != diag.size()) {
        throw std::runtime_error("add_diag_hessian expects a single maintained diagonal block");
    }
    auto diag_block = aligned_vector_map_t(block.diag_panels_.front().data_);
    add_diag_hessian(diag_block, diag, scale);
}

void add_sparse_block(sparse_mat &block, const matrix &delta) {
    if (delta.size() == 0) {
        return;
    }
    if (block.is_empty()) {
        block.insert(0, 0, static_cast<size_t>(delta.rows()), static_cast<size_t>(delta.cols()), sparsity::dense);
    }
    matrix dense(block.rows(), block.cols());
    dense.setZero();
    block.dump_into(dense, spmm::dump_config{.overwrite = true});
    dense.noalias() += delta;
    block = dense;
}
} // namespace

vector gather_lambda_eq(const ns_riccati::ns_riccati_data &d) {
    return gather_field_dual(d, __eq_x, __eq_xu);
}

vector gather_lambda_ineq(const ns_riccati::ns_riccati_data &d) {
    return gather_field_dual(d, __ineq_x, __ineq_xu);
}

bool should_reset_multiplier(const vector_const_ref &multiplier, scalar_t threshold) {
    return threshold <= 0. ||
           (multiplier.size() > 0 && multiplier.cwiseAbs().maxCoeff() > threshold);
}

void maybe_reset_multiplier(vector_ref multiplier, scalar_t threshold, scalar_t reset_value) {
    if (should_reset_multiplier(multiplier, threshold)) {
        multiplier.setConstant(reset_value);
    }
}

void reset_equality_duals(ns_riccati::ns_riccati_data &d, scalar_t threshold) {
    auto lambda_eq = gather_lambda_eq(d);
    if (!should_reset_multiplier(lambda_eq, threshold)) {
        return;
    }
    d.dense_->dual_[__eq_x].setZero();
    d.dense_->dual_[__eq_xu].setZero();
}

void scatter_lambda_eq(ns_riccati::ns_riccati_data &d, const vector_const_ref &lambda) {
    scatter_field_dual(d, __eq_x, __eq_xu, lambda, false);
}

void scatter_lambda_eq_step(ns_riccati::ns_riccati_data &d, const vector_const_ref &delta_lambda) {
    scatter_field_dual(d, __eq_x, __eq_xu, delta_lambda, true);
}

void scatter_lambda_ineq(ns_riccati::ns_riccati_data &d, const vector_const_ref &lambda) {
    scatter_field_dual(d, __ineq_x, __ineq_xu, lambda, false);
}

void scatter_lambda_ineq_step(ns_riccati::ns_riccati_data &d, const vector_const_ref &delta_lambda) {
    scatter_field_dual(d, __ineq_x, __ineq_xu, delta_lambda, true);
}

void prepare_current_constraint_stack(ns_riccati::ns_riccati_data &d) {
    auto *aux = get_aux(d);
    if (aux == nullptr) {
        return;
    }
    d.update_projected_dynamics_residual();
    d.nsp_.s_c_stacked_0_k.conservativeResize(d.ncstr);
    Eigen::Index offset = 0;
    if (d.ns > 0) {
        d.nsp_.s_0_p_k.conservativeResize(d.ns);
        d.nsp_.s_0_p_k.noalias() = d.dense_->approx_[__eq_x].v_;
        d.s_y.times<false>(d.F_0, d.nsp_.s_0_p_k);
        d.nsp_.s_c_stacked_0_k.head(static_cast<Eigen::Index>(d.ns)) = d.nsp_.s_0_p_k;
        offset += static_cast<Eigen::Index>(d.ns);
    }
    if (d.nc > 0) {
        d.nsp_.s_c_stacked_0_k.segment(offset, static_cast<Eigen::Index>(d.nc)) =
            d.dense_->approx_[__eq_xu].v_;
    }
    aux->g_current = gather_raw_ineq_values(d);
}

void initialize_stage(ns_riccati::ns_riccati_data &d) {
    auto *aux = get_aux(d);
    if (aux == nullptr || aux->initialized) {
        return;
    }

    aux->elastic_eq.resize(d.ns, d.nc);
    for (Eigen::Index i = 0; i < aux->elastic_eq.p.size(); ++i) {
        const auto init = initialize_elastic_pair(d.nsp_.s_c_stacked_0_k(i), aux->rho_eq, aux->mu_bar);
        aux->elastic_eq.p(i) = init.p;
        aux->elastic_eq.n(i) = init.n;
        aux->elastic_eq.nu_p(i) = init.z_p;
        aux->elastic_eq.nu_n(i) = init.z_n;
    }
    if (aux->elastic_eq.dim() > 0) {
        vector lambda_eq(aux->elastic_eq.dim());
        for (Eigen::Index i = 0; i < lambda_eq.size(); ++i) {
            const auto init = initialize_elastic_pair(d.nsp_.s_c_stacked_0_k(i), aux->rho_eq, aux->mu_bar);
            lambda_eq(i) = init.lambda;
        }
        scatter_lambda_eq(d, lambda_eq);
        aux->elastic_eq.backup_trial_state();
    }

    aux->elastic_ineq.resize(d.dense_->approx_[__ineq_x].v_.size(), d.dense_->approx_[__ineq_xu].v_.size());
    if (aux->elastic_ineq.dim() > 0) {
        aux->elastic_ineq.t = (-aux->g_current).cwiseMax(1.);
        vector vd = aux->g_current + aux->elastic_ineq.t;
        vector lambda_d(vd.size());
        for (Eigen::Index i = 0; i < vd.size(); ++i) {
            aux->elastic_ineq.nu_t(i) = aux->mu_bar / aux->elastic_ineq.t(i);
            const auto init = initialize_elastic_pair(vd(i), aux->rho_ineq, aux->mu_bar);
            aux->elastic_ineq.p(i) = init.p;
            aux->elastic_ineq.n(i) = init.n;
            aux->elastic_ineq.nu_p(i) = init.z_p;
            aux->elastic_ineq.nu_n(i) = init.z_n;
            lambda_d(i) = init.lambda;
        }
        scatter_lambda_ineq(d, lambda_d);
        aux->elastic_ineq.backup_trial_state();
    }

    aux->initialized = true;
}

void assemble_resto_base_problem(ns_riccati::ns_riccati_data &d,
                                 bool update_derivatives,
                                 scalar_t rho_u,
                                 scalar_t rho_y,
                                 aligned_vector_map_t *u_diag,
                                 aligned_vector_map_t *y_diag) {
    auto *aux = get_aux(d);
    if (aux == nullptr) {
        return;
    }

    d.update_projected_dynamics();
    prepare_current_constraint_stack(d);

    d.dense_->cost_ = 0.;
    d.dense_->lag_ = 0.;
    for (auto pf : primal_fields) {
        d.dense_->cost_jac_[pf].setZero();
        d.dense_->lag_jac_[pf].setZero();
        d.dense_->lag_jac_corr_[pf].setZero();
    }

    const std::array prox_fields{__u, __y};
    const std::array<scalar_t, 2> prox_rho{rho_u, rho_y};
    const std::array<const vector *, 2> prox_ref{&aux->u_ref, &aux->y_ref};
    const std::array<const vector *, 2> prox_sigma{&aux->sigma_u_sq, &aux->sigma_y_sq};
    const std::array<aligned_vector_map_t *, 2> prox_diag{u_diag, y_diag};

    for (size_t i = 0; i < prox_fields.size(); ++i) {
        const auto pf = prox_fields[i];
        const scalar_t rho = prox_rho[i];
        const auto &ref = *prox_ref[i];
        const auto &sigma_sq = *prox_sigma[i];
        if (!(rho > 0.) || ref.size() == 0) {
            continue;
        }
        const vector delta = d.sym_->value_[pf] - ref;
        const scalar_t value = scalar_t(0.5) * rho * sigma_sq.dot(delta.cwiseProduct(delta));
        d.dense_->cost_ += value;
        d.dense_->lag_ += value;
        add_diag_quadratic(d.dense_->cost_jac_[pf], sigma_sq, delta, rho);
        add_diag_quadratic(d.dense_->lag_jac_[pf], sigma_sq, delta, rho);
        if (update_derivatives) {
            if (prox_diag[i] != nullptr && prox_diag[i]->data() != nullptr) {
                add_diag_hessian(*prox_diag[i], sigma_sq, rho);
            } else {
                add_diag_hessian(d.dense_->lag_hess_[pf][pf], sigma_sq, rho);
            }
        }
    }

    for (auto f : std::array{__dyn, __eq_x, __eq_xu}) {
        if (d.full_data_->problem().dim(f) == 0) {
            continue;
        }
        d.dense_->lag_ += d.dense_->approx_[f].v_.dot(d.dense_->dual_[f]);
        for (auto pf : primal_fields) {
            if (!d.dense_->approx_[f].jac_[pf].is_empty()) {
                d.dense_->approx_[f].jac_[pf].right_T_times(d.dense_->dual_[f], d.dense_->lag_jac_[pf]);
            }
        }
    }

    if (aux->g_current.size() > 0) {
        const vector lambda_d = gather_lambda_ineq(d);
        d.dense_->lag_ += aux->g_current.dot(lambda_d);
        for (auto pf : primal_fields) {
            if (!d.dense_->approx_[__ineq_x].jac_[pf].is_empty()) {
                d.dense_->approx_[__ineq_x].jac_[pf].right_T_times(d.dense_->dual_[__ineq_x], d.dense_->lag_jac_[pf]);
            }
            if (!d.dense_->approx_[__ineq_xu].jac_[pf].is_empty()) {
                d.dense_->approx_[__ineq_xu].jac_[pf].right_T_times(d.dense_->dual_[__ineq_xu], d.dense_->lag_jac_[pf]);
            }
        }
    }

    if (aux->elastic_eq.dim() > 0) {
        compute_local_model(d.nsp_.s_c_stacked_0_k, gather_lambda_eq(d), aux->elastic_eq,
                            aux->rho_eq, aux->mu_bar, aux->lambda_reg);
        d.nsp_.s_c_stacked.conservativeResize(d.ncstr, Eigen::NoChange);
        d.nsp_.s_c_stacked.setZero();
        if (d.ns > 0) {
            d.nsp_.s_u.setZero();
            d.s_y.times<false>(d.F_u, d.nsp_.s_u);
            d.nsp_.s_c_stacked.topRows(d.ns) = d.nsp_.s_u;
        }
        if (d.nc > 0) {
            d.c_u.dump_into(d.nsp_.s_c_stacked.bottomRows(d.nc), spmm::dump_config{.overwrite = true});
        }

        d.nsp_.s_c_stacked_0_K.conservativeResize(d.ncstr, d.nx);
        d.nsp_.s_c_stacked_0_K.setZero();
        if (d.ns > 0) {
            d.nsp_.s_0_p_K.conservativeResize(d.ns, d.nx);
            d.nsp_.s_0_p_K.setZero();
            d.s_x.dump_into(d.nsp_.s_0_p_K);
            d.s_y.times<false>(d.F_x, d.nsp_.s_0_p_K);
            d.nsp_.s_c_stacked_0_K.topRows(d.ns) = d.nsp_.s_0_p_K;
        }
        if (d.nc > 0) {
            d.c_x.dump_into(d.nsp_.s_c_stacked_0_K.bottomRows(d.nc));
        }

        const vector eta = aux->elastic_eq.minv_bc;
        d.dense_->lag_jac_corr_[__u].noalias() += (eta.transpose() * d.nsp_.s_c_stacked);
        d.dense_->lag_jac_corr_[__x].noalias() += (eta.transpose() * d.nsp_.s_c_stacked_0_K);
        if (update_derivatives) {
            add_sparse_block(d.dense_->hessian_modification_[__u][__u],
                             d.nsp_.s_c_stacked.transpose() * aux->elastic_eq.minv_diag.asDiagonal() * d.nsp_.s_c_stacked);
            add_sparse_block(d.dense_->hessian_modification_[__u][__x],
                             d.nsp_.s_c_stacked.transpose() * aux->elastic_eq.minv_diag.asDiagonal() * d.nsp_.s_c_stacked_0_K);
            add_sparse_block(d.dense_->hessian_modification_[__x][__x],
                             d.nsp_.s_c_stacked_0_K.transpose() * aux->elastic_eq.minv_diag.asDiagonal() * d.nsp_.s_c_stacked_0_K);
        }
    }

    if (aux->elastic_ineq.dim() > 0) {
        matrix G_u, G_x;
        build_ineq_stack(d, G_u, G_x);
        compute_local_model(aux->g_current, gather_lambda_ineq(d), aux->elastic_ineq,
                            aux->rho_ineq, aux->mu_bar, aux->lambda_reg);
        const vector eta = aux->elastic_ineq.minv_bd;
        d.dense_->lag_jac_corr_[__u].noalias() += (eta.transpose() * G_u);
        d.dense_->lag_jac_corr_[__x].noalias() += (eta.transpose() * G_x);
        if (update_derivatives) {
            add_sparse_block(d.dense_->hessian_modification_[__u][__u],
                             G_u.transpose() * aux->elastic_ineq.minv_diag.asDiagonal() * G_u);
            add_sparse_block(d.dense_->hessian_modification_[__u][__x],
                             G_u.transpose() * aux->elastic_ineq.minv_diag.asDiagonal() * G_x);
            add_sparse_block(d.dense_->hessian_modification_[__x][__x],
                             G_x.transpose() * aux->elastic_ineq.minv_diag.asDiagonal() * G_x);
        }
    }
}

void finalize_newton_step(ns_riccati::ns_riccati_data &d) {
    auto *aux = get_aux(d);
    if (aux == nullptr || !aux->initialized) {
        return;
    }

    if (aux->elastic_eq.dim() > 0) {
        vector delta_c(d.ncstr);
        delta_c.noalias() = d.nsp_.s_c_stacked * d.trial_prim_step[__u];
        delta_c.noalias() += d.nsp_.s_c_stacked_0_K * d.trial_prim_step[__x];
        recover_local_step(delta_c, aux->elastic_eq, aux->lambda_reg);
        scatter_lambda_eq_step(d, aux->elastic_eq.d_lambda);
        d.d_lbd_s_c = aux->elastic_eq.d_lambda;
        if (d.ns > 0) {
            d.s_y.T_times<false>(d.trial_dual_step[__eq_x], d.d_lbd_f);
        }
    }

    if (aux->elastic_ineq.dim() > 0) {
        matrix G_u, G_x;
        build_ineq_stack(d, G_u, G_x);
        aux->g_step.noalias() = G_u * d.trial_prim_step[__u];
        aux->g_step.noalias() += G_x * d.trial_prim_step[__x];
        recover_local_step(aux->g_step, aux->elastic_ineq, aux->lambda_reg);
        scatter_lambda_ineq_step(d, aux->elastic_ineq.d_lambda);
    }
}

void update_ls_bounds(ns_riccati::ns_riccati_data &d, workspace_data *cfg) {
    auto *aux = get_aux(d);
    if (aux == nullptr || !aux->initialized) {
        return;
    }
    auto &ls = cfg->as<linesearch_config>();
    aux->elastic_eq.update_ls_bounds(ls);
    aux->elastic_ineq.update_ls_bounds(ls);
}

void backup_trial_state(ns_riccati::ns_riccati_data &d) {
    auto *aux = get_aux(d);
    if (aux == nullptr || !aux->initialized) {
        return;
    }
    aux->elastic_eq.backup_trial_state();
    aux->elastic_ineq.backup_trial_state();
}

void restore_trial_state(ns_riccati::ns_riccati_data &d) {
    auto *aux = get_aux(d);
    if (aux == nullptr || !aux->initialized) {
        return;
    }
    aux->elastic_eq.restore_trial_state();
    aux->elastic_ineq.restore_trial_state();
}

void apply_affine_step(ns_riccati::ns_riccati_data &d, workspace_data *cfg) {
    auto *aux = get_aux(d);
    if (aux == nullptr || !aux->initialized) {
        return;
    }
    const auto &ls = cfg->as<linesearch_config>();
    aux->elastic_eq.apply_affine_step(ls);
    aux->elastic_ineq.apply_affine_step(ls);

    const scalar_t alpha_eq = ls.dual_alpha_for_eq();
    const scalar_t alpha_resto_dual = ls.dual_alpha_for_ineq();
    const scalar_t alpha_delta = alpha_resto_dual - alpha_eq;
    if (alpha_delta != 0.) {
        for (auto f : std::array{__eq_x, __eq_xu}) {
            if (d.trial_dual_step[f].size() > 0) {
                d.dense_->dual_[f].noalias() += alpha_delta * d.trial_dual_step[f];
            }
        }
    }
}

} // namespace moto::solver::restoration
