#include <moto/solver/restoration/resto_runtime.hpp>

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
    aux->elastic.update_ls_bounds(ls);
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
    aux->elastic.apply_affine_step(ls);
}

} // namespace moto::solver::restoration
