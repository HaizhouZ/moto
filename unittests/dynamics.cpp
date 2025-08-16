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

    explicit_euler dyn2(explicit_euler_impl("test_dynamics2"));
    dyn2.create_2nd_ord_vars("robot2", 18); // todo: for dynamics, check if symbols conflict
    dyn2.add_dt(dt);

    auto prob = ocp::create();
    prob->add(dyn);
    prob->add(dyn2);
    bool show = false;
    size_t N_trials = 100;
    while (N_trials--) {
        auto s_data = sym_data(prob.get());
        auto m_data = merit_data(prob.get());
        auto sh_data = shared_data(prob.get(), &s_data);

        auto d_ptr = dyn->create_approx_data(s_data, m_data, sh_data);
        auto &d = d_ptr->as<explicit_euler_impl::approx_data>();
        auto d2_ptr = dyn2->create_approx_data(s_data, m_data, sh_data);
        auto &d2 = d2_ptr->as<explicit_euler_impl::approx_data>();

        s_data.value_[__x].setRandom();
        s_data.value_[__u].setRandom();
        s_data.value_[__y].setRandom();
        auto &dual = m_data.dual_[__dyn].setRandom();

        dyn->compute_approx(d, true, true, false);
        dyn2->compute_approx(d2, true, true, false);

        array_type<matrix, primal_fields> jac;
        auto merit_jac = m_data.jac_;

        // size_t n_trials = 1;
        if (show) {
            show = false;
            fmt::print("Function value: {}\n", d.v_.transpose());
            fmt::print("Function jacobian:\n");
            for (auto f : primal_fields) {
                auto &jac_sp = m_data.approx_[__dyn].jac_[f];
                fmt::print("{}:\n{}\n", f, jac[f] = jac_sp.dense());
            }
        }
        for (auto f : primal_fields) {
            auto &jac_sp = m_data.approx_[__dyn].jac_[f];
            jac[f] = jac_sp.dense();
            jac_sp.right_T_times(dual, merit_jac[f]);
            REQUIRE(merit_jac[f].isApprox(dual.transpose() * jac[f])); // Merit jacobian does not match the expected value
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

TEST_CASE("dynamics_conflicts") {
    using namespace moto;
    auto [x, y] = sym::states("x", 2);
    auto [x2, y2] = sym::states("x2", 2);

    // test that the symbols conflict
    constr d1("d1", approx_order::first, 2, __dyn);
    d1->add_arguments({x, y});
    constr d2("d2", approx_order::first, 2, __dyn);
    d2->add_arguments({x2, y2});
    d1->add_argument(x2);

    bool conflict_detected = false;
    auto prob = ocp::create();
    try {
        prob->add(d1);
        prob->add(d2);
    } catch (const std::exception &e) {
        fmt::print("Caught expected exception: {}\n", e.what());
        conflict_detected = true;
    }
    REQUIRE(conflict_detected); // Expected a conflict exception but none was thrown
}

TEST_CASE("dynamics_order") {
    using namespace moto;
    auto [x, y] = sym::states("x", 2);
    auto [x2, y2] = sym::states("x2", 2);

    // test that the symbols conflict
    constr d1("d1", approx_order::first, 2, __dyn);
    d1->add_arguments({x, y});
    constr d2("d2", approx_order::first, 2, __dyn);
    d2->add_arguments({x2, y2});
    auto prob = ocp::create();
    prob->add(x2);
    REQUIRE(prob->num(__x) == 1); // Number of expressions in field __x should be 1
    prob->add(d1);
    prob->add(d2);
    for (auto f : primal_fields) {
        std::vector<shared_expr> args;
        args.reserve(prob->num(f));
        args.insert(args.end(), d1->in_args(f).begin(), d1->in_args(f).end());
        args.insert(args.end(), d2->in_args(f).begin(), d2->in_args(f).end());
        auto &exprs = prob->exprs(f);
        REQUIRE(args.size() == exprs.size()); // Number of arguments does not match number of expressions
        for (size_t idx = 0; idx < args.size(); ++idx) {
            REQUIRE(exprs[idx]->uid() == args[idx]->uid()); // Expression UID does not match argument UID - order maybe wrong
        }
    }
}