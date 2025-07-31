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

void print_sym(ns_node_data *cur) {
    auto p = cur->sym_->prob_;
    for (auto f : concat_fields(primal_fields, std::array{__p, __usr_var})) {
        if (p->dim(f) == 0)
            continue; // skip empty fields
        fmt::println("Field {}: dim {}", field::name(f), p->dim(f));
        for (const sym &s : p->exprs(f)) {
            fmt::println("{}: dim {} value {}", s.name(), s.dim(), cur->sym_->get(s).transpose());
        }
    }
}
void print_debug(ns_node_data *cur) {
    print_sym(cur);
    fmt::println("F_u: \n{}", cur->nsp_->F_u);
    fmt::println("F_0_k: \n{}", cur->nsp_->F_0_k.transpose());
    fmt::println("F_0_K: \n{}", cur->nsp_->F_0_K);
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