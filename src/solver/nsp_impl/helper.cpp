#include <Eigen/Eigenvalues>
#include <moto/ocp/impl/sym_data.hpp>
#include <moto/ocp/problem.hpp>
#include <moto/solver/ns_riccati/ns_riccati_data.hpp>
#include <moto/solver/ns_riccati/nullspace_data.hpp>

namespace moto {
namespace solver {
namespace ns_riccati {
void kkt_diagnosis(ns_node_data *cur) {
    auto &d = *cur;
    fmt::print("U is not positive definite\n");
    fmt::print("Eigenvalues of U: \n{}\n", d.nsp_->U.eigenvalues().transpose());
    fmt::print("Eigenvalues of Q_yy: \n{}\n", d.Q_yy.eigenvalues().transpose());
    /// @todo some more maybe about constraints
}


void print_debug(ns_node_data *cur) {
    cur->sym_->print();
    fmt::println("F_u: \n{}", cur->F_u.dense());
    fmt::println("F_0_k: \n{}", cur->F_0.transpose());
    fmt::println("F_0_K: \n{}", cur->F_x.dense());
    fmt::println("F_x: \n{}", cur->dense_->approx_[__dyn].jac_[__x].dense());
    fmt::println("F_y: \n{}", cur->dense_->approx_[__dyn].jac_[__y].dense());
    fmt::println("F_u: \n{}", cur->dense_->approx_[__dyn].jac_[__u].dense());
    fmt::println("u_y_k: \n{}", cur->nsp_->u_y_k.transpose());
    fmt::println("u_y_K: \n{}", cur->nsp_->u_y_K);
    fmt::println("s_c_stacked: \n{}", cur->nsp_->s_c_stacked);
    fmt::println("s_c_stacked_0_k: \n{}", cur->nsp_->s_c_stacked_0_k.transpose());
    fmt::println("s_c_stacked_0_K: \n{}", cur->nsp_->s_c_stacked_0_K);
    fmt::print("rank: {} of {} equality constraints\n", cur->nsp_->rank, cur->ncstr);
}
} // namespace ns_riccati
} // namespace solver
} // namespace moto