#define MOTO_NS_RICCATI_IMPL

#include <moto/solver/ns_riccati/generic_solver.hpp>
#include <moto/utils/field_conversion.hpp>

namespace moto {
namespace solver {
namespace ns_riccati {
namespace {
void propagate_value(ns_riccati_data &d, ns_riccati_data *prev,
                     bool hessian) {
    if (prev == nullptr)
        return;
    auto &perm = utils::permutation_from_y_to_x(prev->dense_->prob_,
                                                d.dense_->prob_);
    d.Q_x *= perm;
    prev->Q_y.noalias() += d.Q_x;
    if (hessian) {
        d.V_xx *= perm;
        d.V_xx.applyOnTheLeft(perm.transpose());
        prev->V_yy.noalias() += d.V_xx;
    }
}
} // namespace

extern void print_debug(ns_riccati_data *cur);

void generic_solver::riccati_recursion(ns_riccati_data *cur,
                                       ns_riccati_data *prev) {
    auto &d = *cur;
    auto &nsp = d.nsp_;
    d.V_yy.array() /= 2;
    d.V_yy = d.V_yy + d.V_yy.transpose().eval();
    if (d.V_yy.hasNaN() || !d.V_yy.allFinite()) {
        print_debug(cur);
        throw std::runtime_error("V_yy has NaN or inf");
    }

    nsp.y_0_p_k.noalias() +=
        d.Q_y.transpose() - d.V_yy * nsp.y_y_k;
    nsp.z_0_k.noalias() += nsp.Z_y.transpose() * nsp.y_0_p_k;
    d.Q_x.noalias() -= nsp.y_0_p_k.transpose() * nsp.y_y_K;
    nsp.Q_zz.noalias() +=
        nsp.Z_y.transpose() * d.V_yy * nsp.Z_y;
    nsp.y_0_p_K.noalias() = d.V_yy * nsp.y_y_K;
    nsp.z_0_K.noalias() -= nsp.Z_y.transpose() * nsp.y_0_p_K;
    d.V_xx.noalias() += nsp.y_y_K.transpose() * nsp.y_0_p_K;

    if (nsp.Q_zz.rows()) {
        nsp.llt_ns_.compute(nsp.Q_zz);
        if (!nsp.llt_ns_.valid())
            throw std::runtime_error(
                "projected primal Hessian is not positive definite");
        nsp.llt_ns_.solve(nsp.z_0_K, nsp.z_K, -1.0);
        d.Q_x.noalias() += nsp.z_0_k.transpose() * nsp.z_K;
        d.V_xx.noalias() += nsp.z_0_K.transpose() * nsp.z_K;
    } else {
        nsp.z_K.resize(0, d.nx);
    }
    propagate_value(d, prev, true);
}

void generic_solver::riccati_recursion_correction(ns_riccati_data *cur,
                                                  ns_riccati_data *prev) {
    auto &d = *cur;
    auto &nsp = d.nsp_;
    if (d.rank_status_ == rank_status::unconstrained)
        nsp.z_0_k = d.Q_u.transpose();
    else
        nsp.z_0_k.noalias() = nsp.Z_u.transpose() * d.Q_u.transpose();
    if (d.uses_sparse_lifted_basis())
        linear_backend::transpose_multiply(
            d.lifting_.l_u(), d.Q_l.transpose(), nsp.z_0_k, -1.);
    else
        nsp.z_0_k.noalias() += nsp.Z_l.transpose() * d.Q_l.transpose();
    d.Q_x.noalias() -= d.Q_u * nsp.u_y_K;
    if (d.uses_sparse_lifted_basis())
        linear_backend::right_multiply(
            d.Q_l, d.lifting_.l_x(), d.Q_x, -1.);
    else
        d.Q_x.noalias() -= d.Q_l * nsp.l_y_K;
    nsp.y_0_p_k = d.Q_y.transpose();
    nsp.z_0_k.noalias() += nsp.Z_y.transpose() * nsp.y_0_p_k;
    d.Q_x.noalias() -= nsp.y_0_p_k.transpose() * nsp.y_y_K;
    if (nsp.Q_zz.rows())
        d.Q_x.noalias() += nsp.z_0_k.transpose() * nsp.z_K;
    propagate_value(d, prev, false);
}
} // namespace ns_riccati
} // namespace solver
} // namespace moto
