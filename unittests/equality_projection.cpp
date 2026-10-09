#include <catch2/catch_test_macros.hpp>
#include <moto/solver/ns_sqp.hpp>
#include <moto/ocp/cost.hpp>
#include <moto/ocp/dynamics/dense_dynamics.hpp>
#include <Eigen/QR>
#include <cstdlib>

using namespace moto;

TEST_CASE("SQP projection backends match horizon KKT with redundant hard equalities") {
    setenv("MOTO_SYNC_CODEGEN", "1", 1);
    auto [x,y] = sym::states("projection_option_x",2);
    auto u = sym::inputs("projection_option_u",2);
    const cs::SX &sx=x, &su=u;
    const cs::SX sum=su(0)+su(1)-1.;
    auto stage=stage_ocp::create();
    stage->add(*dynamics(new dense_dynamics("projection_option_dyn",y-x-u,approx_order::first)));
    stage->add(*cost(new generic_cost("projection_option_cost",var_inarg_list{},
        .25*cs::SX::dot(sx,sx)+.5*su(0)*su(0)+su(1)*su(1),approx_order::second)));
    stage->add(*generic_constr::create("projection_option_equalities",{},
        cs::SX::vertcat({sum,2.*sum,cs::SX(0)}),approx_order::first));
    ns_sqp sqp(2);
    REQUIRE(sqp.settings.equality_projection == ns_sqp::equality_projection_backend::eigen);
    for (int k=0;k<3;++k) sqp.stages().push_back(stage->copy());
    sqp.ed().add(*cost(new generic_cost("projection_option_terminal",var_inarg_list{},
        .25*cs::SX::dot(sx,sx),approx_order::second)));
    sqp.settings.restoration.enabled=false;
    sqp.settings.regularization.validate_direction=true;
    sqp.settings.prim_tol=sqp.settings.dual_tol=1e-9;

    matrix h=matrix::Zero(12,12), a=matrix::Zero(9,12);
    for (int k=0;k<3;++k) {
        h.diagonal().segment(4*k,4) << 1.,2.,.5,.5;
        a.block<2,2>(2*k,4*k)=-matrix::Identity(2,2);
        a.block<2,2>(2*k,4*k+2)=matrix::Identity(2,2);
        if(k) a.block<2,2>(2*k,4*(k-1)+2)=-matrix::Identity(2,2);
        a(6+k,4*k)=a(6+k,4*k+1)=1.;
    }
    matrix kkt=matrix::Zero(21,21);
    kkt.topLeftCorner(12,12)=h;
    kkt.topRightCorner(12,9)=a.transpose();
    kkt.bottomLeftCorner(9,12)=a;
    vector rhs=vector::Zero(21); rhs.tail(3).setOnes();
    const vector expected=kkt.colPivHouseholderQr().solve(rhs);
    REQUIRE((kkt*expected-rhs).norm()<1e-12);

    // Reuse one realized graph while switching in both directions. Equality
    // initialization and iterative-refinement overlays use the same setting.
    for (auto backend : {ns_sqp::equality_projection_backend::eigen,
                         ns_sqp::equality_projection_backend::panel_lu,
                         ns_sqp::equality_projection_backend::eigen}) {
        sqp.settings.equality_projection=backend;
        for(auto *node : sqp.solver_nodes())
            for(auto field : primal_fields) node->sym_val().value_[field].setZero();
        sqp.update(1,false);
        REQUIRE(sqp.linear_solve_last.status==ns_sqp::linear_solve_status::success);
        REQUIRE(sqp.linear_solve_last.regularization==0.);
        REQUIRE(sqp.linear_solve_last.stationarity_residual<1e-10);
        REQUIRE(sqp.linear_solve_last.equality_residual<1e-10);
        for(int k=0;k<3;++k) {
            auto *node=sqp.solver_nodes()[k];
            REQUIRE((node->sym_val().value_[__u]-expected.segment(4*k,2)).norm()<1e-9);
            REQUIRE((node->sym_val().value_[__y]-expected.segment(4*k+2,2)).norm()<1e-9);
        }
    }
}
