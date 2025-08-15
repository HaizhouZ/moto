#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <iostream>
#include <moto/ocp/dynamics/euler_dynamics.hpp>
#include <moto/ocp/problem.hpp>
#define ENABLE_TIMED_BLOCK
#include <moto/utils/timed_block.hpp>
#include <thread>
TEST_CASE("dynamics") {
    using namespace moto;
    explicit_euler dyn(explicit_euler_impl("test_dynamics"));
    // dyn.create_2nd_ord_vars("robot", 3);
    // dyn.create_1st_ord_vars("foot", 6);
    // dyn.create_1st_ord_vars("hand", 6);
    // dyn.create_2nd_ord_vars("obj", 3);
    dyn.create_2nd_ord_vars("robot", 18);
    // dyn.add_dt(0.1);
    var dt = sym::inputs("dt", 1);
    dyn.add_dt(dt);

    auto prob = ocp::create();
    prob->add(dyn);
    size_t N_trials = 100;
    while (N_trials--) {
        auto s_data = sym_data(prob.get());
        auto m_data = merit_data(prob.get());
        auto sh_data = shared_data(prob.get(), &s_data);

        auto d_ptr = dyn->create_approx_data(s_data, m_data, sh_data);
        auto &d = d_ptr->as<explicit_euler_impl::approx_data>();

        s_data.value_[__x].setRandom();
        s_data.value_[__u].setRandom();
        s_data.value_[__y].setRandom();
        auto &dual = m_data.dual_[__dyn].setRandom();

        dyn->compute_approx(d, true, true, false);

        array_type<matrix, primal_fields> jac;
        auto merit_jac = m_data.jac_;

        fmt::print("Function value: {}\n", d.v_.transpose());
        fmt::print("Function jacobian:\n");
        // size_t n_trials = 1;
        for (auto f : primal_fields) {
            auto &jac_sp = m_data.approx_[__dyn].jac_[f];
            jac_sp.right_T_times(dual, merit_jac[f]);
            fmt::print("{}:\n{}\n", f, jac[f] = jac_sp.dense());
            assert(merit_jac[f].isApprox(dual.transpose() * jac[f]) && "Merit jacobian does not match the expected value");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        size_t n_trials = 100;

        timed_block_labeled(
            "sparse",
            auto n = n_trials;
            while (n--) {
                for (auto f : primal_fields) {
                    // auto f = __x;
                    auto &jac_sp = m_data.approx_[__dyn].jac_[f];
                    jac_sp.right_T_times(dual, merit_jac[f]);
                }
            });
        timed_block_labeled(
            "dense",
            auto n = n_trials;
            while (n--) {
                for (auto f : primal_fields) {
                    // auto f = __x;
                    auto &jac_sp = m_data.approx_[__dyn].jac_[f];
                    merit_jac[f].noalias() = dual.transpose() * jac[f];
                    // assert(merit_jac[f].isApprox(dual.transpose() * jac[f]) && "Merit jacobian does not match the expected value");
                }
            });
    }
}