#include <moto/solver/restoration/resto_runtime.hpp>

#include <moto/ocp/impl/node_data.hpp>
#include <moto/ocp/impl/sym_data.hpp>
#include <moto/solver/ns_sqp.hpp>
#include <moto/solver/restoration/resto_local_kkt.hpp>

#include <fmt/format.h>

namespace moto::solver::restoration {

namespace {
auto *get_aux(ns_riccati::ns_riccati_data &d) {
    return dynamic_cast<ns_riccati::ns_riccati_data::restoration_aux_data *>(d.aux_.get());
}

const auto *get_aux(const ns_riccati::ns_riccati_data &d) {
    return dynamic_cast<const ns_riccati::ns_riccati_data::restoration_aux_data *>(d.aux_.get());
}

auto &get_resto_prox(ns_riccati::ns_riccati_data &d) {
    return static_cast<ns_sqp::data &>(d).restoration_prox_;
}

const auto &get_resto_prox(const ns_riccati::ns_riccati_data &d) {
    return static_cast<const ns_sqp::data &>(d).restoration_prox_;
}
} // namespace

vector gather_lambda(const ns_riccati::ns_riccati_data &d) {
    vector lambda(static_cast<Eigen::Index>(d.ncstr));
    Eigen::Index offset = 0;
    if (d.ns > 0) {
        lambda.head(static_cast<Eigen::Index>(d.ns)) = d.dense_->dual_[__eq_x];
        offset += static_cast<Eigen::Index>(d.ns);
    }
    if (d.nc > 0) {
        lambda.tail(static_cast<Eigen::Index>(d.nc)) = d.dense_->dual_[__eq_xu];
    } else if (offset < lambda.size()) {
        lambda.tail(lambda.size() - offset).setZero();
    }
    return lambda;
}

void scatter_lambda(ns_riccati::ns_riccati_data &d, const vector_const_ref &lambda) {
    if (lambda.size() != static_cast<Eigen::Index>(d.ncstr)) {
        throw std::runtime_error("scatter_lambda size mismatch");
    }
    if (d.ns > 0) {
        d.dense_->dual_[__eq_x] = lambda.head(static_cast<Eigen::Index>(d.ns));
    }
    if (d.nc > 0) {
        d.dense_->dual_[__eq_xu] = lambda.tail(static_cast<Eigen::Index>(d.nc));
    }
}

void scatter_lambda_step(ns_riccati::ns_riccati_data &d, const vector_const_ref &delta_lambda) {
    if (delta_lambda.size() != static_cast<Eigen::Index>(d.ncstr)) {
        throw std::runtime_error("scatter_lambda_step size mismatch");
    }
    if (d.ns > 0) {
        d.trial_dual_step[__eq_x] = delta_lambda.head(static_cast<Eigen::Index>(d.ns));
    } else {
        d.trial_dual_step[__eq_x].resize(0);
    }
    if (d.nc > 0) {
        d.trial_dual_step[__eq_xu] = delta_lambda.tail(static_cast<Eigen::Index>(d.nc));
    } else {
        d.trial_dual_step[__eq_xu].resize(0);
    }
}

void prepare_current_constraint_stack(ns_riccati::ns_riccati_data &d) {
    if (d.ncstr == 0) {
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
}

void initialize_stage(ns_riccati::ns_riccati_data &d) {
    auto *aux = get_aux(d);
    if (aux == nullptr || aux->initialized || d.ncstr == 0) {
        if (aux != nullptr && d.ncstr == 0) {
            aux->initialized = true;
        }
        return;
    }

    aux->elastic.resize(d.ns, d.nc);

    vector lambda(static_cast<Eigen::Index>(d.ncstr));
    for (Eigen::Index i = 0; i < lambda.size(); ++i) {
        const auto init = initialize_elastic_pair(d.nsp_.s_c_stacked_0_k(i), aux->rho_eq, aux->mu_bar);
        aux->elastic.p(i) = init.p;
        aux->elastic.n(i) = init.n;
        aux->elastic.nu_p(i) = init.z_p;
        aux->elastic.nu_n(i) = init.z_n;
        lambda(i) = init.lambda;
    }
    aux->elastic.backup_trial_state();
    scatter_lambda(d, lambda);
    aux->initialized = true;
}

void assemble_resto_base_lagrangian(ns_riccati::ns_riccati_data &d) {
    d.dense_->lag_ = 0.;
    for (auto field : primal_fields) {
        d.dense_->lag_jac_[field].setZero();
    }
    for (auto f : hard_constr_fields) {
        if (d.full_data_->problem().dim(f) == 0) {
            continue;
        }
        d.dense_->lag_ += d.dense_->approx_[f].v_.dot(d.dense_->dual_[f]);
        for (auto p : primal_fields) {
            if (d.dense_->approx_[f].jac_[p].is_empty()) {
                continue;
            }
                d.dense_->approx_[f].jac_[p].right_T_times(d.dense_->dual_[f], d.dense_->lag_jac_[p]);
        }
    }
    for (auto f : std::array{__eq_x, __eq_xu}) {
        if (d.full_data_->problem().dim(f) == 0) {
            continue;
        }
        d.dense_->lag_ += d.dense_->approx_[f].v_.dot(d.dense_->dual_[f]);
        for (auto p : primal_fields) {
            if (d.dense_->approx_[f].jac_[p].is_empty()) {
                continue;
            }
            d.dense_->approx_[f].jac_[p].right_T_times(d.dense_->dual_[f], d.dense_->lag_jac_[p]);
        }
    }
}

void add_resto_prox_term(ns_riccati::ns_riccati_data &d, scalar_t rho_u, scalar_t rho_y) {
    auto *aux = get_aux(d);
    if (aux == nullptr) {
        return;
    }
    const auto &prox = get_resto_prox(d);

    // Restoration owns this proximal term as a special w=(x,u,y)-only cost.
    // It is part of the base restoration Lagrangian itself, not a reduced
    // Schur-complement correction, so it updates only the base lagrangian
    // value/derivatives. Outer objective bookkeeping (cost_, cost_jac_) must
    // remain that of the original NLP.
    if (prox.u_ref.size() > 0) {
        const auto &u = d.sym_->value_[__u];
        const vector du = u - prox.u_ref;
        d.dense_->lag_ +=
            scalar_t(0.5) * rho_u * prox.sigma_u_sq.dot(du.cwiseProduct(du));
        d.dense_->lag_jac_[__u].noalias() +=
            rho_u * (prox.sigma_u_sq.array() * du.array()).matrix().transpose();
        auto &diag_panels = d.dense_->lag_hess_[__u][__u].diag_panels_;
        if (!diag_panels.empty()) {
            diag_panels.back().data_.array() += rho_u * prox.sigma_u_sq.array();
        }
    }
    if (prox.y_ref.size() > 0) {
        const auto &y = d.sym_->value_[__y];
        const vector dy = y - prox.y_ref;
        d.dense_->lag_ +=
            scalar_t(0.5) * rho_y * prox.sigma_y_sq.dot(dy.cwiseProduct(dy));
        d.dense_->lag_jac_[__y].noalias() +=
            rho_y * (prox.sigma_y_sq.array() * dy.array()).matrix().transpose();
        auto &diag_panels = d.dense_->lag_hess_[__y][__y].diag_panels_;
        if (!diag_panels.empty()) {
            diag_panels.back().data_.array() += rho_y * prox.sigma_y_sq.array();
        }
    }
}

void finalize_newton_step(ns_riccati::ns_riccati_data &d) {
    auto *aux = get_aux(d);
    if (aux == nullptr || !aux->initialized || d.ncstr == 0) {
        return;
    }

    const vector lambda = gather_lambda(d);
    vector delta_c(static_cast<Eigen::Index>(d.ncstr));
    delta_c.noalias() = d.nsp_.s_c_stacked * d.trial_prim_step[__u];
    delta_c.noalias() += d.nsp_.s_c_stacked_0_K * d.trial_prim_step[__x];

    const auto current_summary = current_local_residuals(aux->elastic);
    recover_local_step(delta_c, aux->elastic, aux->lambda_reg);
    const auto recovered_summary = recovered_linearized_residuals(delta_c, aux->elastic, aux->lambda_reg);
    const auto predicted_summary =
        predicted_trial_residuals(delta_c, lambda, aux->elastic, aux->rho_eq, aux->mu_bar);

    if (aux->verbose) {
        fmt::print("[resto local] ncstr={} current: c={:.3e} stat=({:.3e},{:.3e}) comp=({:.3e},{:.3e}) | "
                   "recovered: c={:.3e} stat=({:.3e},{:.3e}) comp=({:.3e},{:.3e}) | "
                   "fullstep-pred: c={:.3e} stat=({:.3e},{:.3e}) comp=({:.3e},{:.3e})\n",
                   d.ncstr,
                   current_summary.inf_c,
                   current_summary.inf_p, current_summary.inf_n,
                   current_summary.inf_s_p, current_summary.inf_s_n,
                   recovered_summary.inf_c,
                   recovered_summary.inf_p, recovered_summary.inf_n,
                   recovered_summary.inf_s_p, recovered_summary.inf_s_n,
                   predicted_summary.inf_c,
                   predicted_summary.inf_p, predicted_summary.inf_n,
                   predicted_summary.inf_s_p, predicted_summary.inf_s_n);
    }

    d.d_lbd_s_c = aux->elastic.d_lambda;
    scatter_lambda_step(d, aux->elastic.d_lambda);
    if (d.ns > 0) {
        d.s_y.T_times<false>(d.trial_dual_step[__eq_x], d.d_lbd_f);
    }
}

void update_ls_bounds(ns_riccati::ns_riccati_data &d, workspace_data *cfg) {
    auto *aux = get_aux(d);
    if (aux == nullptr || !aux->initialized || d.ncstr == 0) {
        return;
    }
    auto &ls = cfg->as<linesearch_config>();
    const scalar_t primal_before = ls.primal.alpha_max;
    const scalar_t dual_before = ls.dual.alpha_max;
    aux->elastic.update_ls_bounds(ls);
    if (aux->verbose) {
        auto describe_bound = [](const char *label, const vector &value, const vector &step) {
            scalar_t alpha = 1.;
            Eigen::Index idx = -1;
            scalar_t value_i = 0.;
            scalar_t step_i = 0.;
            for (Eigen::Index i = 0; i < value.size(); ++i) {
                if (step(i) < 0.) {
                    const scalar_t cand = -0.995 * value(i) / step(i);
                    if (cand < alpha) {
                        alpha = cand;
                        idx = i;
                        value_i = value(i);
                        step_i = step(i);
                    }
                }
            }
            return std::tuple<const char *, scalar_t, Eigen::Index, scalar_t, scalar_t>{label, alpha, idx, value_i, step_i};
        };

        const auto [p_label, p_alpha, p_idx, p_val, p_step] = describe_bound("p", aux->elastic.p, aux->elastic.d_p);
        const auto [n_label, n_alpha, n_idx, n_val, n_step] = describe_bound("n", aux->elastic.n, aux->elastic.d_n);
        const auto [zp_label, zp_alpha, zp_idx, zp_val, zp_step] = describe_bound("nu_p", aux->elastic.nu_p, aux->elastic.d_nu_p);
        const auto [zn_label, zn_alpha, zn_idx, zn_val, zn_step] = describe_bound("nu_n", aux->elastic.nu_n, aux->elastic.d_nu_n);

        if (ls.primal.alpha_max < primal_before || ls.dual.alpha_max < dual_before) {
            fmt::print("[resto bounds] ncstr={} primal {:.3e}->{:.3e} dual {:.3e}->{:.3e} | "
                       "{}[{}]: value {:.3e}, step {:.3e}, alpha {:.3e} | "
                       "{}[{}]: value {:.3e}, step {:.3e}, alpha {:.3e} | "
                       "{}[{}]: value {:.3e}, step {:.3e}, alpha {:.3e} | "
                       "{}[{}]: value {:.3e}, step {:.3e}, alpha {:.3e}\n",
                       d.ncstr,
                       primal_before, ls.primal.alpha_max,
                       dual_before, ls.dual.alpha_max,
                       p_label, p_idx, p_val, p_step, p_alpha,
                       n_label, n_idx, n_val, n_step, n_alpha,
                       zp_label, zp_idx, zp_val, zp_step, zp_alpha,
                       zn_label, zn_idx, zn_val, zn_step, zn_alpha);
        }
    }
}

void backup_trial_state(ns_riccati::ns_riccati_data &d) {
    auto *aux = get_aux(d);
    if (aux == nullptr || !aux->initialized || d.ncstr == 0) {
        return;
    }
    aux->elastic.backup_trial_state();
}

void restore_trial_state(ns_riccati::ns_riccati_data &d) {
    auto *aux = get_aux(d);
    if (aux == nullptr || !aux->initialized || d.ncstr == 0) {
        return;
    }
    aux->elastic.restore_trial_state();
}

void apply_affine_step(ns_riccati::ns_riccati_data &d, workspace_data *cfg) {
    auto *aux = get_aux(d);
    if (aux == nullptr || !aux->initialized || d.ncstr == 0) {
        return;
    }
    const auto &ls = cfg->as<linesearch_config>();
    const scalar_t alpha_eq = ls.dual_alpha_for_eq();
    const scalar_t alpha_resto_dual = ls.dual_alpha_for_ineq();
    aux->elastic.apply_affine_step(ls);
    // generic_solver::apply_affine_step() has already advanced all hard
    // equality multipliers using alpha_eq. The explicit elastic local block,
    // however, couples lambda_c with (nu_p, nu_n) in the stationarity rows
    // r_p = rho - lambda_c - nu_p and r_n = rho + lambda_c - nu_n.
    // To keep that local dual block coherent during restoration line search,
    // lambda_c must use the same dual step size as (nu_p, nu_n), i.e. the
    // inequality/dual alpha. Adjust the scattered eq multipliers here without
    // touching the dynamics multiplier __dyn.
    const scalar_t alpha_delta = alpha_resto_dual - alpha_eq;
    if (alpha_delta != 0.) {
        if (d.trial_dual_step[__eq_x].size() > 0) {
            d.dense_->dual_[__eq_x].noalias() += alpha_delta * d.trial_dual_step[__eq_x];
        }
        if (d.trial_dual_step[__eq_xu].size() > 0) {
            d.dense_->dual_[__eq_xu].noalias() += alpha_delta * d.trial_dual_step[__eq_xu];
        }
    }
}

} // namespace moto::solver::restoration
