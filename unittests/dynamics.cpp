#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <iostream>
#include <moto/ocp/dynamics/euler_dynamics.hpp>
#include <moto/ocp/problem.hpp>

TEST_CASE("dynamics") {
    using namespace moto;
    explicit_euler dyn(explicit_euler_impl("test_dynamics"));
    dyn.create_2nd_ord_vars("robot", 3);
    dyn.create_1st_ord_vars("foot", 6);
    dyn.create_1st_ord_vars("hand", 6);
    dyn.create_2nd_ord_vars("obj", 3);
    dyn.add_dt(0.1);

    auto prob = ocp::create();
    prob->add(dyn);

    auto s_data = sym_data(prob.get());
    auto m_data = merit_data(prob.get());
    auto sh_data = shared_data(prob.get(), &s_data);

    auto d_ptr = dyn->create_approx_data(s_data, m_data, sh_data);
    auto& d = d_ptr->as<explicit_euler_impl::approx_data>();

    s_data.value_[__x].setRandom();
    s_data.value_[__u].setRandom();
    s_data.value_[__y].setRandom();

    dyn->compute_approx(d, true, true, false);

    fmt::print("Function value: {}\n", d.v_.transpose());
    fmt::print("Function jacobian:\n");
    fmt::print("x:\n{}\n", m_data.approx_[__dyn].jac_[__x].dense());
    fmt::print("u:\n{}\n", m_data.approx_[__dyn].jac_[__u].dense());
    fmt::print("y:\n{}\n", m_data.approx_[__dyn].jac_[__y].dense());
}