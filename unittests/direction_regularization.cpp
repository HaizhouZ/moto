#include <catch2/catch_test_macros.hpp>
#include <moto/solver/ns_sqp.hpp>
#include <moto/ocp/dynamics/dense_dynamics.hpp>
#include <moto/ocp/cost.hpp>
#ifdef MOTO_TEST_MULTIBODY
#include <moto/multibody/casadi_manifold.hpp>
#endif
#include <moto/solver/restoration/resto_overlay.hpp>
#include <Eigen/QR>
#include <cstdlib>

using namespace moto;
namespace {
const bool sync_codegen = [] { setenv("MOTO_SYNC_CODEGEN", "1", 1); return true; }();
cost objective(const std::string &name, const cs::SX &value) {
    return cost(new generic_cost(name, var_inarg_list{}, value, approx_order::second));
}
struct fixture {
    var x, y, u;
    cs::SX sx, su;
    stage_ocp_ptr_t stage = stage_ocp::create();
    ns_sqp sqp{1};
    fixture(const std::string &name) {
        std::tie(x, y) = sym::states(name + "_x", 1);
        u = sym::inputs(name + "_u", 1);
        sx = static_cast<const cs::SX &>(x); su = static_cast<const cs::SX &>(u);
        stage->add(*dynamics(new dense_dynamics(name + "_dyn", y-x-u, approx_order::first)));
        sqp.settings.eq_init.enabled = false;
        sqp.settings.restoration.enabled = false;
    }
    void start(size_t horizon = 1) {
        for (size_t k = 0; k < horizon; ++k) sqp.stages().push_back(stage->copy());
        for (auto *d : sqp.solver_nodes())
            for (auto f : primal_fields) d->sym_val().value_[f].setZero();
    }
};
}

TEST_CASE("Riccati Newton direction matches independent horizon KKT solution") {
    fixture f("reference_direction");
    f.stage->add(*objective("reference_running", .5*f.sx*f.sx + f.su*f.su));
    f.sqp.ed().add(*objective("reference_terminal", 2.5*(f.sx-1.)*(f.sx-1.)));
    f.start(3);
    // Independent full-space KKT in [u0,x1,u1,x2,u2,x3].
    matrix H = matrix::Zero(6, 6);
    H.diagonal() << 2., 1., 2., 1., 2., 5.;
    matrix A = matrix::Zero(3, 6);
    A << -1., 1., 0., 0., 0., 0., 0., -1., -1., 1., 0., 0., 0., 0., 0., -1., -1., 1.;
    matrix K = matrix::Zero(9, 9);
    K.topLeftCorner(6,6) = H; K.topRightCorner(6,3) = A.transpose(); K.bottomLeftCorner(3,6) = A;
    vector rhs = vector::Zero(9); rhs(5)=5.;
    const vector reference = K.colPivHouseholderQr().solve(rhs);
    REQUIRE((K*reference-rhs).norm() < 1e-12);
    f.sqp.update(1, false);
    REQUIRE(f.sqp.linear_solve_last.status == ns_sqp::linear_solve_status::success);
    REQUIRE(f.sqp.linear_solve_last.regularization == 0.);
    const auto &nodes = f.sqp.solver_nodes();
    for (size_t k=0; k<3; ++k) {
        REQUIRE(std::abs(nodes[k]->sym_val().value_[__u](0)-reference(2*k)) < 1e-10);
        REQUIRE(std::abs(nodes[k]->sym_val().value_[__y](0)-reference(2*k+1)) < 1e-10);
    }
}

TEST_CASE("Negative curvature triggers consistent full-space primal regularization") {
    fixture f("negative_curvature");
    f.stage->add(*objective("negative_cost", -.5*f.su*f.su + .25*f.su*f.su*f.su*f.su));
    f.start();
    auto *d = f.sqp.solver_nodes()[0];
    d->sym_val().value_[__u](0)=.1; d->sym_val().value_[__y](0)=.1;
    f.sqp.update(1, false);
    const auto &info = f.sqp.linear_solve_last;
    REQUIRE(info.status == ns_sqp::linear_solve_status::success);
    REQUIRE(info.attempts > 1);
    REQUIRE(info.regularization > 0.);
    // Regularizes u and y, with dy=du; the reduced shift must be 2*delta.
    const scalar_t expected = .1 + .099 / (-.97 + 2.*info.regularization);
    REQUIRE(std::abs(d->sym_val().value_[__u](0)-expected) < 1e-9);
    REQUIRE(std::abs(d->sym_val().value_[__y](0)-expected) < 1e-9);
}

TEST_CASE("Consistent duplicate equalities retain their feasible direction") {
    fixture f("duplicate_eq");
    f.stage->add(*objective("duplicate_cost", .5*f.su*f.su));
    f.stage->add(*generic_constr::create("duplicate_rows", {}, cs::SX::vertcat({f.su-1., 2.*f.su-2.}), approx_order::first));
    f.start();
    f.sqp.update(1, false);
    REQUIRE(f.sqp.linear_solve_last.status == ns_sqp::linear_solve_status::success);
    REQUIRE(std::abs(f.sqp.solver_nodes()[0]->sym_val().value_[__u](0)-1.) < 1e-10);
}

TEST_CASE("Inconsistent or unactuated equalities cannot be silently dropped") {
    fixture f("inconsistent_eq");
    f.stage->add(*objective("inconsistent_cost", .5*f.su*f.su));
    SECTION("inconsistent duplicate input rows") {
        f.stage->add(*generic_constr::create("inconsistent_rows", {}, cs::SX::vertcat({f.su-1., f.su-2.}), approx_order::first));
    }
    SECTION("fixed initial state equality") {
        f.stage->add(*generic_constr::create("unactuated_row", {}, f.sx-1., approx_order::first));
    }
    f.start();
    const auto result = f.sqp.update(1, false);
    REQUIRE(result.iter.result == ns_sqp::iter_result_t::numerical_failure);
    REQUIRE(f.sqp.linear_solve_last.status == ns_sqp::linear_solve_status::inconsistent_equalities);
    REQUIRE(f.sqp.linear_solve_last.attempts == 1);
    for (auto f0 : primal_fields) REQUIRE(f.sqp.solver_nodes()[0]->sym_val().value_[f0].isZero());
}

TEST_CASE("Exhausted curvature retries preserve the accepted state") {
    fixture f("bounded_retries");
    f.stage->add(*objective("bounded_negative_cost", -.5*f.su*f.su));
    f.start();
    f.sqp.settings.regularization.maximum = 1e-3;
    const auto result = f.sqp.update(1, false);
    REQUIRE(result.iter.result == ns_sqp::iter_result_t::numerical_failure);
    REQUIRE(f.sqp.linear_solve_last.status == ns_sqp::linear_solve_status::factorization_failed);
    REQUIRE(f.sqp.linear_solve_last.attempts <= 3);
    for (auto f0 : primal_fields) REQUIRE(f.sqp.solver_nodes()[0]->sym_val().value_[f0].isZero());
}

TEST_CASE("Mixed input-next-state curvature retains full KKT stationarity") {
    fixture f("mixed_curvature");
    const cs::SX &y = f.y;
    f.stage->add(*objective("mixed_curvature_cost", .5*f.su*f.su + .5*y*y + .3*f.su*y - y));
    f.start();
    f.sqp.update(1, false);
    REQUIRE(f.sqp.linear_solve_last.status == ns_sqp::linear_solve_status::success);
    REQUIRE(f.sqp.linear_solve_last.regularization == 0.);
    REQUIRE(std::abs(f.sqp.solver_nodes()[0]->sym_val().value_[__u](0) - 1./2.6) < 1e-10);
}

TEST_CASE("Full-space residual includes exact nonlinear constraint Hessians") {
    fixture f("exact_constraint_curvature");
    f.stage->add(*objective("exact_constraint_cost", .5*f.su*f.su));
    f.stage->add(*generic_constr::create("exact_constraint", {}, f.su*f.su-1., approx_order::second));
    f.start();
    auto *d = f.sqp.solver_nodes()[0];
    d->sym_val().value_[__u](0) = .8;
    d->sym_val().value_[__y](0) = .8;
    d->dense().dual_[__eq_xu](0) = .7;
    f.sqp.update(1, false);
    REQUIRE(f.sqp.linear_solve_last.status == ns_sqp::linear_solve_status::success);
    REQUIRE(f.sqp.linear_solve_last.regularization == 0.);
    REQUIRE(f.sqp.linear_solve_last.stationarity_residual < 1e-12);
    REQUIRE(d->dense().has_constraint_hessian_);
}

#ifdef MOTO_TEST_MULTIBODY
TEST_CASE("Restoration proximity uses tangent dimensions for manifold states") {
    using namespace solver::restoration;
    const auto q = cs::SX::sym("resto_circle_q", 2);
    const auto v = cs::SX::sym("resto_circle_v", 1);
    const auto other = cs::SX::sym("resto_circle_other", 2);
    const auto angle = cs::SX::atan2(q(1), q(0));
    auto [x, y] = multibody::casadi_manifold::create("resto_circle", q, v,
        cs::SX::vertcat({cs::SX::cos(angle+v), cs::SX::sin(angle+v)}), other,
        angle - cs::SX::atan2(other(1), other(0)), (vector(2) << 1., 0.).finished());
    auto u = sym::inputs("resto_circle_u", 1);
    auto source = ocp::create();
    source->add(*dynamics(new dense_dynamics("resto_circle_dyn",
        y->symbolic_difference(y, x)-u, approx_order::first)));
    source->wait_until_ready();
    auto overlay = build_restoration_overlay_problem(source,
        restoration_overlay_settings{.rho_u=.1, .rho_y=.25});
    node_data outer(source), restored(overlay);
    scalar_t mu = 1.;
    sync_outer_to_restoration_state(outer, restored, 1e-4, &mu);
    REQUIRE(restored.problem().dim(__y) == 2);
    REQUIRE(restored.problem().tdim(__y) == 1);
    restored.for_each(__cost, [&](const resto_prox_cost &, resto_prox_cost::approx_data &d) {
        REQUIRE(d.y_ref.size() == 2);
        REQUIRE(d.sigma_y_sq.size() == 1);
        REQUIRE(d.sigma_y_sq(0) == .25);
    });
    restored.update_approximation(node_data::update_mode::eval_all);
    REQUIRE(std::isfinite(restored.dense().cost_));
    REQUIRE(restored.dense().lag_hess_[__y][__y].dense().allFinite());
}

#endif
