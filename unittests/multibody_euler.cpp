#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>

#include <moto/ocp/problem.hpp>
#define ENABLE_TIMED_BLOCK
#include <moto/multibody/stacked_euler.hpp>
#include <moto/utils/timed_block.hpp>

#include <Eigen/LU>

TEST_CASE("multibody_euler") {
    using namespace moto;
    using dyn_t = multibody::stacked_euler;
    auto dt = sym::inputs("dt", 1);
    func dyn = dyn_t("stacked_euler");
    auto e = multibody::euler::from_urdf("rsc/go2_description/urdf/go2_description.urdf", dt);
    auto e2 = multibody::euler::from_urdf("rsc/iiwa_description/urdf/iiwa14.urdf", dt,
                                          multibody::root_joint_t::fixed,
                                          multibody::euler::v_int_type::_explicit);
    auto e3 = multibody::euler::from_urdf("rsc/iiwa_description/urdf/iiwa14.urdf", dt,
                                          multibody::root_joint_t::fixed,
                                          multibody::euler::v_int_type::_mid_point);
    dyn.as<dyn_t>().add(e);
    dyn.as<dyn_t>().add(e2);
    dyn.as<dyn_t>().add(e3);
    dyn.as<dyn_t>().add(e.as<multibody::euler>().share());

    auto prob = ocp::create();
    prob->add(dyn);

    prob->print_summary();

    auto s_data = sym_data(prob.get());
    auto m_data = merit_data(prob.get());
    auto sh_data = shared_data(prob.get(), &s_data);

    auto d_ptr = dyn->create_approx_data(s_data, m_data, sh_data);
    auto &d = *d_ptr;

    s_data.value_[__x].setRandom();
    s_data.value_[__u].setRandom();
    s_data.value_[__y].setRandom();
    auto &dual = m_data.dual_[__dyn].setRandom();

    dyn->compute_approx(d, true, true, false);
    // dyn2->compute_approx(d2, true, true, false);

    dyn.as<dyn_t>().compute_project_derivatives(d);

    // fmt::println("f_x dense:\n{:.1}", m_data.approx_[__dyn].jac_[__x].dense());
    // fmt::println("f_y dense:\n{:.1}", m_data.approx_[__dyn].jac_[__y].dense());
    // fmt::println("f_u dense:\n{:.1}", m_data.approx_[__dyn].jac_[__u].dense());
    // fmt::println("proj_f_x dense:\n{}", m_data.proj_f_x().dense());
    // fmt::println("proj_f_u dense:\n{}", m_data.proj_f_u().dense());
    auto lu_ = m_data.approx_[__dyn].jac_[__y].dense().lu();
    {
        vector ground_truth = lu_.solve(m_data.approx_[__dyn].v_);
        REQUIRE(m_data.proj_f_res().isApprox(ground_truth));
    }
    {
        matrix ground_truth = lu_.solve(m_data.approx_[__dyn].jac_[__x].dense());
        // fmt::println("groudn_Truth:\n{}", ground_truth);
        // fmt::println("residual:\n{}", m_data.proj_f_x().dense() - ground_truth);
        REQUIRE(m_data.proj_f_x().dense().isApprox(ground_truth));
    }
    {
        matrix ground_truth = lu_.solve(m_data.approx_[__dyn].jac_[__u].dense());
        // fmt::println("groudn_Truth:\n{}", ground_truth);
        // fmt::println("residual:\n{}", m_data.proj_f_u().dense() - ground_truth);
        REQUIRE(m_data.proj_f_u().dense().isApprox(ground_truth));
    }
}