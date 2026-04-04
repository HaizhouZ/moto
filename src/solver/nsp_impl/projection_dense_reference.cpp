#include <moto/solver/projection/dense_reference.hpp>
#include <moto/solver/ns_riccati/ns_riccati_data.hpp>

namespace moto::solver::projection {

namespace {

rank_case to_rank_case(ns_riccati::rank_status status) {
    switch (status) {
    case ns_riccati::rank_status::unconstrained:
        return rank_case::unconstrained;
    case ns_riccati::rank_status::constrained:
        return rank_case::constrained;
    case ns_riccati::rank_status::fully_constrained:
        return rank_case::fully_constrained;
    }
    return rank_case::inconsistent;
}

void capture_common(const ns_riccati::ns_riccati_data &d, dense_snapshot &snapshot) {
    auto &out = snapshot.out;
    auto &nsp = d.nsp_;

    d.default_elimination_stage_.export_problem(d, snapshot.prob);

    out.info.rank_status = to_rank_case(d.rank_status_);
    out.info.numeric_rank = nsp.rank;
    out.info.nullity = nsp.Z_u.cols() > 0 ? static_cast<size_t>(nsp.Z_u.cols())
                                          : (d.rank_status_ == ns_riccati::rank_status::unconstrained ? d.nu : 0);
    out.info.structurally_reusable = true;
}

} // namespace

void snapshot_dense_factorization(const ns_riccati::ns_riccati_data &d,
                                  dense_snapshot &snapshot) {
    snapshot.clear_numeric();
    capture_common(d, snapshot);

    auto &out = snapshot.out;
    auto &nsp = d.nsp_;

    out.has_factorization = true;
    out.U_p = nsp.u_y_K;
    out.u_p = nsp.u_y_k;
    out.Y_p = nsp.y_y_K;
    out.y_p = nsp.y_y_k;
    out.T_u = nsp.Z_u;
    out.T_y = nsp.Z_y;
}

void snapshot_dense_reduced_step(const ns_riccati::ns_riccati_data &d,
                                 dense_snapshot &snapshot) {
    snapshot_dense_factorization(d, snapshot);

    auto &out = snapshot.out;
    auto &nsp = d.nsp_;

    out.has_reduced_step = true;
    out.K_red = nsp.z_K;
    out.k_red = nsp.z_k;
}

void snapshot_dense_duals(const ns_riccati::ns_riccati_data &d,
                          dense_snapshot &snapshot) {
    snapshot_dense_reduced_step(d, snapshot);
    snapshot.out.has_duals = true;
    snapshot.out.dlbd_proj = d.d_lbd_s_c;
}

} // namespace moto::solver::projection
