#include <double_integrator/cost.hpp>
#include <double_integrator/dynamics.hpp>
#include <iostream>
#include <moto/solver/ns_riccati/nullspace_data.hpp>
#include <moto/solver/ns_sqp.hpp>
#include <moto/utils/print.hpp>

using namespace moto;

// void consistency_check() {
//     struct test_data {
//         doubleIntegratorDyn dyn;
//         sym_data_ptr_t s_data;
//         merit_data_ptr_t m_data;
//         shared_data_ptr_t s_shared;
//         ocp_ptr_t prob;
//         func_approx_data_ptr_t data;
//         matrix D, innerproduct;
//         test_data(bool dense) : dyn(dense), D(6, 6), innerproduct(6, 6) {
//             innerproduct.setZero();
//             prob = ocp::create();
//             prob->add(dyn);
//             s_data.reset(new sym_data(prob.get()));
//             m_data.reset(new merit_data(prob.get()));
//             s_shared.reset(new shared_data(prob.get(), s_data.get()));
//             data = dyn.dyn->create_approx_data(*s_data, *m_data, *s_shared);
//             m_data->dual_[__dyn].setConstant(33);
//         }
//         void set_random() {
//             s_data->get(dyn.r).setRandom();
//             s_data->get(dyn.v).setRandom();
//             s_data->get(dyn.a).setRandom();
//             s_data->get(dyn.r_next).setRandom();
//             s_data->get(dyn.v_next).setRandom();
//             // m_data->dual_[__dyn].setRandom();
//             D.setRandom();
//             D = D * D.transpose(); // make it positive definite
//         }
//         void copy(const test_data &other) {
//             s_data->get(dyn.r) = other.s_data->get(other.dyn.r);
//             s_data->get(dyn.v) = other.s_data->get(other.dyn.v);
//             s_data->get(dyn.a) = other.s_data->get(other.dyn.a);
//             s_data->get(dyn.r_next) = other.s_data->get(other.dyn.r_next);
//             s_data->get(dyn.v_next) = other.s_data->get(other.dyn.v_next);
//             D = other.D;
//         }
//         void run() {
//             dyn.dyn->compute_approx(*data, true, true);
//             dyn.dyn.as<generic_dynamics>().compute_project_derivatives(*data);
//             dyn.dyn.as<generic_dynamics>().apply_jac_y_inverse_transpose(*data, data->v_, m_data->dual_[__dyn]);
//             // innerproduct.setZero();
//             for (auto f : primal_fields) {
//                 m_data->approx_[__dyn].jac_[f].right_T_times(m_data->dual_[__dyn], m_data->jac_[f]);
//             }
//             m_data->proj_f_u().inner_product(D, innerproduct);
//         }
//     } dense(true), sparse(false);
//     size_t n = 10000;
//     while (n--) {
//         dense.set_random();
//         sparse.copy(dense);
//         dense.run();
//         sparse.run();

//         assert(dense.m_data->approx_[__dyn].v_.isApprox(sparse.m_data->approx_[__dyn].v_));
//         for (auto f : primal_fields) {
//             assert(dense.m_data->approx_[__dyn].jac_[f].dense().isApprox(sparse.m_data->approx_[__dyn].jac_[f].dense()));
//             assert(dense.m_data->jac_[f].isApprox(sparse.m_data->jac_[f]));
//         }
//         assert(dense.m_data->proj_f_x().dense().isApprox(sparse.m_data->proj_f_x().dense()));
//         assert(dense.m_data->proj_f_u().dense().isApprox(sparse.m_data->proj_f_u().dense()));
//         assert(dense.m_data->proj_f_res().isApprox(sparse.m_data->proj_f_res()));
//         assert(dense.m_data->dual_[__dyn].isApprox(sparse.m_data->dual_[__dyn]));
//         assert(dense.innerproduct.isApprox(sparse.innerproduct));
//     }
// }

int main() {
    // consistency_check();
    // return 0;
    doubleIntegratorDyn dyn(false);
    doubleIntegratorCosts costs;
    auto prob = ocp::create();
    prob->add(dyn);
    prob->add(costs.running(dyn.r, dyn.v, dyn.a));
    auto prob_terminal = prob->clone();
    prob_terminal->add(costs.terminal(dyn.r_next, dyn.v_next));

    utils::print_problem(prob);
    utils::print_problem(prob_terminal);

    fmt::print("good!\n");

    ns_sqp sqp;

    auto &init_node = sqp.graph_.add(sqp.create_node(prob));
    auto &end_node = sqp.graph_.add(sqp.create_node(prob_terminal));

    sqp.graph_.add_edge(init_node, end_node, 100);

    sqp.graph_.set_head(init_node);
    sqp.graph_.set_tail(end_node);

    init_node->value(dyn.r).setConstant(6);
    init_node->value(dyn.v).setZero();

    sqp.graph_.apply_forward([&dyn](auto *cur, auto *next) {
        next->value(dyn.r) = cur->value(dyn.r);
        cur->value(dyn.r_next) = next->value(dyn.r);
    });

    size_t n_iter = 100;

    sqp.update(n_iter);
    // per step timing
    // for (size_t i = 0; i < sqp.timings.size(); i++) {
    //     fmt::print("Timing[{}]: {}us\n", i, sqp.timings[i] / n_iter);
    // }

    // auto &data = ns_riccati::get_data(init_node.get());
    // std::cout << data.rollout_->prim_[__x].transpose() << '\n';
    // sqp.update();
    sqp.graph_.apply_forward([&dyn](auto *node) {
        // std::cout << "delX  " << data.rollout_->prim_[__x].transpose() << '\n';
        // std::cout << field::name(data.rank_status_) << '\n';
        // std::cout << "state " << node->value(__x).transpose() << '\n';
        // std::cout << "input " << node->value(__u).transpose() << '\n';
        // std::cout << "inf_prim_res: " << node->inf_prim_res_ << '\n';
        // std::cout << "nexts " << data.sym_->value_[__y].transpose() << '\n';
        // std::cout << "rescs " << node->data(dyn.vel_zero_constr).v_.transpose() << '\n';
        // std::cout << "dual  " << static_cast<constr_approx_data &>(node->data(dyn.vel_zero_constr)).multiplier_.transpose() << '\n';
        // std::cout << "sy    " << data.nsp_->s_y << '\n';
        // std::cout << "su    " << data.nsp_->s_u << '\n';
        //     std::cout << "resdy " << data.dense_->approx_[__dyn].v_.transpose() << '\n';
        //     std::cout << "dual  " << data.dense_->dual_[__dyn].transpose() << '\n';
        //     std::cout << "Qx    " << data.Q_x << '\n';
        //     std::cout << "Qu    " << data.Q_u << '\n';
        //     std::cout << "Qy    " << data.Q_y << '\n';
        //     // std::cout << "gain \n" << data.d_y.k.transpose() << '\n';
        //     // std::cout << data.d_u.K.transpose() << '\n';
        //     // std::cout << data.nsp_->u_0_p_k.transpose() << '\n';
        //     // std::cout << data.nsp_->U << '\n';
        //     // std::cout << data.Q_y.transpose() << '\n' << '\n';
        //     // std::cout << data.Q_yx << '\n' << '\n';
        //     // std::cout << data.dense_->approx_[__dyn].v_[__x] << '\n' << '\n';
        // std::cout << '\n';
    });

    fmt::print("nice!\n");

    return 0;
}