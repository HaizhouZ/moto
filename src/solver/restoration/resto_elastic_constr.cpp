#include <moto/solver/restoration/resto_elastic_constr.hpp>

#include <algorithm>

namespace moto::solver {

namespace {
scalar_t positivity_alpha_max(const vector &value, const vector &step, scalar_t fraction_to_boundary) {
    scalar_t alpha = 1.;
    for (Eigen::Index i = 0; i < value.size(); ++i) {
        if (step(i) < 0.) {
            alpha = std::min(alpha, -fraction_to_boundary * value(i) / step(i));
        }
    }
    return std::max(alpha, scalar_t(0.));
}
} // namespace

void resto_elastic_constr::resize(size_t ns_dim, size_t nc_dim) {
    ns = ns_dim;
    nc = nc_dim;
    const auto dim_eig = static_cast<Eigen::Index>(dim());
    p.resize(dim_eig);
    p_backup.resize(dim_eig);
    d_p.resize(dim_eig);
    nu_p.resize(dim_eig);
    nu_p_backup.resize(dim_eig);
    d_nu_p.resize(dim_eig);
    n.resize(dim_eig);
    n_backup.resize(dim_eig);
    d_n.resize(dim_eig);
    nu_n.resize(dim_eig);
    nu_n_backup.resize(dim_eig);
    d_nu_n.resize(dim_eig);
    c_current.resize(dim_eig);
    r_c.resize(dim_eig);
    r_p.resize(dim_eig);
    r_n.resize(dim_eig);
    r_s_p.resize(dim_eig);
    r_s_n.resize(dim_eig);
    combo_p.resize(dim_eig);
    combo_n.resize(dim_eig);
    b_c.resize(dim_eig);
    minv_diag.resize(dim_eig);
    minv_bc.resize(dim_eig);
    d_lambda.resize(dim_eig);

    p.setZero();
    p_backup.setZero();
    d_p.setZero();
    nu_p.setZero();
    nu_p_backup.setZero();
    d_nu_p.setZero();
    n.setZero();
    n_backup.setZero();
    d_n.setZero();
    nu_n.setZero();
    nu_n_backup.setZero();
    d_nu_n.setZero();
    c_current.setZero();
    r_c.setZero();
    r_p.setZero();
    r_n.setZero();
    r_s_p.setZero();
    r_s_n.setZero();
    combo_p.setZero();
    combo_n.setZero();
    b_c.setZero();
    minv_diag.setZero();
    minv_bc.setZero();
    d_lambda.setZero();
}

void resto_elastic_constr::backup_trial_state() {
    p_backup = p;
    nu_p_backup = nu_p;
    n_backup = n;
    nu_n_backup = nu_n;
}

void resto_elastic_constr::restore_trial_state() {
    p = p_backup;
    nu_p = nu_p_backup;
    n = n_backup;
    nu_n = nu_n_backup;
}

void resto_elastic_constr::apply_affine_step(const linesearch_config &cfg) {
    if (p.size() > 0) {
        p.noalias() += cfg.alpha_primal * d_p;
        n.noalias() += cfg.alpha_primal * d_n;
        nu_p.noalias() += cfg.dual_alpha_for_ineq() * d_nu_p;
        nu_n.noalias() += cfg.dual_alpha_for_ineq() * d_nu_n;
    }
}

void resto_elastic_constr::update_ls_bounds(linesearch_config &cfg, scalar_t fraction_to_boundary) const {
    if (p.size() == 0) {
        return;
    }
    cfg.primal.alpha_max = std::min(cfg.primal.alpha_max,
                                    positivity_alpha_max(p, d_p, fraction_to_boundary));
    cfg.primal.alpha_max = std::min(cfg.primal.alpha_max,
                                    positivity_alpha_max(n, d_n, fraction_to_boundary));
    cfg.dual.alpha_max = std::min(cfg.dual.alpha_max,
                                  positivity_alpha_max(nu_p, d_nu_p, fraction_to_boundary));
    cfg.dual.alpha_max = std::min(cfg.dual.alpha_max,
                                  positivity_alpha_max(nu_n, d_nu_n, fraction_to_boundary));
}

} // namespace moto::solver
