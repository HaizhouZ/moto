#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <filesystem>

#include <moto/ocp/constr.hpp>
#include <moto/ocp/cost.hpp>
#include <moto/ocp/dynamics/dense_dynamics.hpp>
#include <moto/ocp/problem.hpp>
#include <moto/solver/projection/default_stage.hpp>
#include <moto/solver/projection/dense_reference.hpp>
#include <moto/solver/ns_sqp.hpp>

namespace {
namespace cs = casadi;

const bool force_sync_codegen_for_test = []() {
    setenv("MOTO_SYNC_CODEGEN", "1", 1);
    std::filesystem::create_directories("gen");
    return true;
}();

moto::vector vec(std::initializer_list<moto::scalar_t> values) {
    moto::vector out(values.size());
    Eigen::Index idx = 0;
    for (moto::scalar_t v : values) {
        out(idx++) = v;
    }
    return out;
}

moto::ocp_ptr_t make_fully_constrained_problem() {
    using namespace moto;
    auto [x, y] = sym::states("x_proj_dense", 1);
    auto u = sym::inputs("u_proj_dense", 1);
    x->set_default_value(vec({1.0}));
    y->set_default_value(vec({4.0}));
    u->set_default_value(vec({5.0}));

    auto prob = ocp::create();
    auto dyn = dynamics(new dense_dynamics("dyn_proj_dense",
                                           var_list{x, y, u},
                                           y - cs::SX(2.0) * x - cs::SX(3.0) * u,
                                           approx_order::second, __dyn));
    auto eq_y = constr(new generic_constr("eq_y_proj_dense", var_list{y}, cs::SX(5.0) * y, approx_order::second, __undefined));
    auto u_cost = cost(new generic_cost("cost_u_proj_dense", var_list{u}, (u * u) * cs::SX(0.5), approx_order::second));
    prob->add(*dyn);
    prob->add(*eq_y);
    prob->add(*u_cost);
    return prob;
}

moto::ocp_ptr_t make_partially_constrained_problem() {
    using namespace moto;
    auto [x, y] = sym::states("x_proj_partial", 1);
    auto u0 = sym::inputs("u0_proj_partial", 1);
    auto u1 = sym::inputs("u1_proj_partial", 1);
    x->set_default_value(vec({1.0}));
    y->set_default_value(vec({3.0}));
    u0->set_default_value(vec({2.0}));
    u1->set_default_value(vec({-1.0}));

    auto prob = ocp::create();
    auto dyn = dynamics(new dense_dynamics("dyn_proj_partial",
                                           var_list{x, y, u0, u1},
                                           y - x - cs::SX(2.0) * u0 + u1,
                                           approx_order::second, __dyn));
    auto eq_y = constr(new generic_constr("eq_y_proj_partial", var_list{y}, cs::SX(3.0) * y, approx_order::second, __undefined));
    auto u_cost = cost(new generic_cost("cost_u_proj_partial",
                                        var_list{u0, u1},
                                        (u0 * u0 + cs::SX(4.0) * u1 * u1) * cs::SX(0.5),
                                        approx_order::second));
    prob->add(*dyn);
    prob->add(*eq_y);
    prob->add(*u_cost);
    return prob;
}

moto::ocp_ptr_t make_xy_projected_state_problem() {
    using namespace moto;
    auto [x, y] = sym::states("x_proj_xy", 1);
    auto u = sym::inputs("u_proj_xy", 1);
    x->set_default_value(vec({1.0}));
    y->set_default_value(vec({4.0}));
    u->set_default_value(vec({5.0}));

    auto prob = ocp::create();
    auto dyn = dynamics(new dense_dynamics("dyn_proj_xy",
                                           var_list{x, y, u},
                                           y - cs::SX(2.0) * x - cs::SX(3.0) * u,
                                           approx_order::second, __dyn));
    auto eq_xy = constr(new generic_constr("eq_xy_proj",
                                           var_list{x, y},
                                           cs::SX(7.0) * x + cs::SX(5.0) * y,
                                           approx_order::second, __eq_x));
    auto u_cost = cost(new generic_cost("cost_u_proj_xy", var_list{u}, (u * u) * cs::SX(0.5), approx_order::second));
    prob->add(*dyn);
    prob->add(*eq_xy);
    prob->add(*u_cost);
    return prob;
}

moto::ocp_ptr_t make_partitioned_mixed_problem() {
    using namespace moto;
    auto [x, y] = sym::states("x_proj_split", 2);
    auto u = sym::inputs("u_proj_split", 2);
    x->set_default_value(vec({1.0, -2.0}));
    y->set_default_value(vec({4.0, 5.0}));
    u->set_default_value(vec({3.0, -1.0}));

    auto prob = ocp::create();
    auto dyn = dynamics(new dense_dynamics("dyn_proj_split",
                                           var_list{x, y, u},
                                           cs::SX::vertcat(std::vector{
                                               y(0) - x(0) - cs::SX(2.0) * u(0) + u(1),
                                               y(1) - cs::SX(3.0) * x(1) + u(0) + cs::SX(4.0) * u(1),
                                           }),
                                           approx_order::second, __dyn));
    auto eq_y = constr(new generic_constr("eq_y_proj_split",
                                          var_list{y},
                                          cs::SX::vertcat(std::vector{
                                              cs::SX(2.0) * y(0),
                                              cs::SX(-3.0) * y(1),
                                          }),
                                          approx_order::second, __undefined));
    auto eq_u = constr(new generic_constr("eq_u_proj_split",
                                          var_list{u},
                                          cs::SX::vertcat(std::vector{
                                              u(0) + cs::SX(2.0) * u(1),
                                              cs::SX(3.0) * u(0) - u(1),
                                          }),
                                          approx_order::second, __undefined));
    prob->add(*dyn);
    prob->add(*eq_y);
    prob->add(*eq_u);
    return prob;
}

moto::ocp_ptr_t make_partially_constrained_mixed_problem() {
    using namespace moto;
    auto [x, y] = sym::states("x_proj_mix_partial", 1);
    auto u0 = sym::inputs("u0_proj_mix_partial", 1);
    auto u1 = sym::inputs("u1_proj_mix_partial", 1);
    auto u2 = sym::inputs("u2_proj_mix_partial", 1);
    x->set_default_value(vec({1.0}));
    y->set_default_value(vec({2.0}));
    u0->set_default_value(vec({3.0}));
    u1->set_default_value(vec({-1.0}));
    u2->set_default_value(vec({0.5}));

    auto prob = ocp::create();
    auto dyn = dynamics(new dense_dynamics("dyn_proj_mix_partial",
                                           var_list{x, y, u0, u1, u2},
                                           y - x - cs::SX(2.0) * u0 + u1 - u2,
                                           approx_order::second, __dyn));
    auto eq_y = constr(new generic_constr("eq_y_proj_mix_partial",
                                          var_list{y},
                                          cs::SX(3.0) * y,
                                          approx_order::second, __undefined));
    auto eq_u = constr(new generic_constr("eq_u_proj_mix_partial",
                                          var_list{u0, u1, u2},
                                          u0 + cs::SX(2.0) * u1 - u2,
                                          approx_order::second, __undefined));
    prob->add(*dyn);
    prob->add(*eq_y);
    prob->add(*eq_u);
    return prob;
}

moto::ocp_ptr_t make_odd_row_split_input_problem() {
    using namespace moto;
    auto [x, y] = sym::states("x_proj_odd_split", 3);
    auto u0 = sym::inputs("u0_proj_odd_split", 1);
    auto u1 = sym::inputs("u1_proj_odd_split", 1);
    x->set_default_value(vec({1.0, -2.0, 0.5}));
    y->set_default_value(vec({4.0, 5.0, -3.0}));
    u0->set_default_value(vec({3.0}));
    u1->set_default_value(vec({-1.0}));

    auto prob = ocp::create();
    auto dyn = dynamics(new dense_dynamics("dyn_proj_odd_split",
                                           var_list{x, y, u0, u1},
                                           cs::SX::vertcat(std::vector{
                                               y(0) - x(0) - u0,
                                               y(1) - x(1) - cs::SX(2.0) * u1,
                                               y(2) - x(2) + cs::SX(3.0) * u0 - cs::SX(4.0) * u1,
                                           }),
                                           approx_order::second, __dyn));
    prob->add(*dyn);
    return prob;
}
} // namespace

TEST_CASE("dense reference snapshot mirrors projected equality assembly") {
    using namespace moto;
    using namespace moto::solver::projection;

    ns_sqp::data stage(make_fully_constrained_problem());
    solver::ns_riccati::generic_solver solver;
    dense_snapshot snapshot;

    stage.update_approximation(node_data::update_mode::eval_all);
    solver.ns_factorization(&stage);

    REQUIRE(snapshot.prob.A_dense.size() == 0);
    REQUIRE(snapshot.out.U_p.size() == 0);

    snapshot_dense_factorization(stage, snapshot);

    const auto &prob = snapshot.prob;
    const auto &out = snapshot.out;
    REQUIRE(out.has_factorization);
    REQUIRE(!out.has_reduced_step);
    REQUIRE(!out.has_duals);

    matrix expected_A(1, 1);
    expected_A << 15.0;
    matrix expected_B(1, 1);
    expected_B << 10.0;
    vector expected_a(1);
    expected_a << 85.0;

    REQUIRE(prob.A_dense.rows() == expected_A.rows());
    REQUIRE(prob.A_dense.cols() == expected_A.cols());
    REQUIRE(prob.B_dense.rows() == expected_B.rows());
    REQUIRE(prob.B_dense.cols() == expected_B.cols());
    REQUIRE(prob.a_dense.size() == expected_a.size());
    CAPTURE(prob.A_dense(0, 0));
    CAPTURE(prob.B_dense(0, 0));
    CAPTURE(prob.a_dense(0));
    REQUIRE(prob.A_dense.isApprox(expected_A));
    REQUIRE(prob.B_dense.isApprox(expected_B));
    REQUIRE(prob.a_dense.isApprox(expected_a));
    REQUIRE(prob.A_dense.isApprox(stage.nsp_.s_c_stacked));
    REQUIRE(prob.B_dense.isApprox(stage.nsp_.s_c_stacked_0_K));
    REQUIRE(prob.a_dense.isApprox(stage.nsp_.s_c_stacked_0_k));
    REQUIRE(prob.rows.size() == 1);
    REQUIRE(prob.rows[0].kind == row_kind::projected_state_eq);
    REQUIRE(prob.rows[0].source_index == 0);
    REQUIRE(prob.rows[0].canonical_index == 0);
    REQUIRE(prob.rows[0].group == "__eq_x_projected");
    REQUIRE(prob.cols.size() == 1);
    REQUIRE(prob.cols[0].kind == column_kind::control);
    REQUIRE(prob.cols[0].source_index == 0);
    REQUIRE(prob.cols[0].canonical_index == 0);
    REQUIRE(prob.cols[0].group == "__u");

    REQUIRE(stage.rank_status_ == solver::ns_riccati::rank_status::fully_constrained);
    REQUIRE(stage.nsp_.Z_u.cols() == 0);
    REQUIRE(out.info.rank_status == rank_case::fully_constrained);
    REQUIRE(out.info.numeric_rank == stage.nsp_.rank);
    REQUIRE(out.U_p.rows() == stage.nsp_.u_y_K.rows());
    REQUIRE(out.U_p.cols() == stage.nsp_.u_y_K.cols());
    REQUIRE(out.u_p.size() == stage.nsp_.u_y_k.size());
    REQUIRE(out.Y_p.rows() == stage.nsp_.y_y_K.rows());
    REQUIRE(out.Y_p.cols() == stage.nsp_.y_y_K.cols());
    REQUIRE(out.y_p.size() == stage.nsp_.y_y_k.size());
    REQUIRE(out.U_p.isApprox(stage.nsp_.u_y_K));
    REQUIRE(out.u_p.isApprox(stage.nsp_.u_y_k));
    REQUIRE(out.Y_p.isApprox(stage.nsp_.y_y_K));
    REQUIRE(out.y_p.isApprox(stage.nsp_.y_y_k));
}

TEST_CASE("compiled default elimination stage assembles projected operator directly") {
    using namespace moto;

    ns_sqp::data stage(make_partially_constrained_problem());
    auto &compiled = stage.default_elimination_stage_;

    stage.update_approximation(node_data::update_mode::eval_all);
    stage.update_projected_dynamics();
    compiled.prepare_factorization_problem(stage);
    compiled.assemble_control_jacobian(stage);
    compiled.assemble_state_jacobian(stage);
    stage.update_projected_dynamics_residual();
    compiled.prepare_residual_problem(stage);
    compiled.assemble_residual(stage);

    REQUIRE(compiled.projected_state_rows == 1);
    REQUIRE(compiled.state_input_rows == 0);
    REQUIRE(compiled.rows.size() == 1);
    REQUIRE(compiled.cols.size() == 2);

    matrix expected_A(1, 2);
    expected_A << 6.0, -3.0;
    matrix expected_B(1, 1);
    expected_B << 3.0;
    vector expected_a(1);
    expected_a << 18.0;

    REQUIRE(stage.nsp_.s_c_stacked.isApprox(expected_A));
    REQUIRE(stage.nsp_.s_c_stacked_0_K.isApprox(expected_B));
    REQUIRE(stage.nsp_.s_c_stacked_0_k.isApprox(expected_a));
}

TEST_CASE("dense dynamics projected Jacobians handle odd-row split input blocks") {
    using namespace moto;

    ns_sqp::data stage(make_odd_row_split_input_problem());

    stage.update_approximation(node_data::update_mode::eval_all);
    REQUIRE_NOTHROW(stage.update_projected_dynamics());

    const matrix proj_u = stage.dense().proj_f_u().dense();
    const matrix raw_u = stage.dense().approx_[__dyn].jac_[__u].dense();
    const matrix proj_x = stage.dense().proj_f_x().dense();
    const matrix raw_x = stage.dense().approx_[__dyn].jac_[__x].dense();

    REQUIRE(proj_u.rows() == 3);
    REQUIRE(proj_u.cols() == 2);
    REQUIRE(proj_x.rows() == 3);
    REQUIRE(proj_x.cols() == 3);
    REQUIRE(proj_u.isApprox(raw_u));
    REQUIRE(proj_x.isApprox(raw_x));
}

TEST_CASE("explicit __eq_x with x and y contributes both direct and projected terms") {
    using namespace moto;
    using namespace moto::solver::projection;

    ns_sqp::data stage(make_xy_projected_state_problem());
    solver::ns_riccati::generic_solver solver;
    dense_snapshot snapshot;

    stage.update_approximation(node_data::update_mode::eval_all);
    solver.ns_factorization(&stage);
    snapshot_dense_factorization(stage, snapshot);

    matrix expected_A(1, 1);
    expected_A << 15.0;
    matrix expected_B(1, 1);
    expected_B << 17.0;
    vector expected_a(1);
    expected_a << 92.0;

    REQUIRE(stage.nsp_.s_c_stacked.isApprox(expected_A));
    REQUIRE(stage.nsp_.s_c_stacked_0_K.isApprox(expected_B));
    REQUIRE(stage.nsp_.s_c_stacked_0_k.isApprox(expected_a));

    REQUIRE(snapshot.prob.A_dense.isApprox(expected_A));
    REQUIRE(snapshot.prob.B_dense.isApprox(expected_B));
    REQUIRE(snapshot.prob.a_dense.isApprox(expected_a));
    REQUIRE(snapshot.prob.rows.size() == 1);
    REQUIRE(snapshot.prob.rows[0].kind == row_kind::projected_state_eq);
    REQUIRE(snapshot.out.info.rank_status == rank_case::fully_constrained);
    REQUIRE(stage.rank_status_ == solver::ns_riccati::rank_status::fully_constrained);
    REQUIRE(stage.nsp_.u_y_K.isApprox(expected_B / expected_A(0, 0)));
    REQUIRE(stage.nsp_.u_y_k.isApprox(expected_a / expected_A(0, 0)));
}

TEST_CASE("compiled default elimination stage supports multiple ordered equality blocks") {
    using namespace moto;
    using namespace moto::solver::projection;

    ns_sqp::data stage(make_partitioned_mixed_problem());
    auto &compiled = stage.default_elimination_stage_;

    stage.update_approximation(node_data::update_mode::eval_all);
    stage.update_projected_dynamics();
    compiled.prepare_factorization_problem(stage);
    compiled.assemble_control_jacobian(stage);
    compiled.assemble_state_jacobian(stage);
    stage.update_projected_dynamics_residual();
    compiled.prepare_residual_problem(stage);
    compiled.assemble_residual(stage);

    const matrix A_default = stage.nsp_.s_c_stacked;
    const matrix B_default = stage.nsp_.s_c_stacked_0_K;
    const vector a_default = stage.nsp_.s_c_stacked_0_k;

    compiled.configure_partitioned(stage.ns, stage.nc, stage.nu, {1, 1}, {1, 1});
    REQUIRE(compiled.blocks.size() == 4);
    REQUIRE(compiled.rows.size() == 4);
    REQUIRE(compiled.rows[0].group == "__eq_x_projected:0");
    REQUIRE(compiled.rows[1].group == "__eq_x_projected:1");
    REQUIRE(compiled.rows[2].group == "__eq_xu:0");
    REQUIRE(compiled.rows[3].group == "__eq_xu:1");

    compiled.prepare_factorization_problem(stage);
    compiled.assemble_control_jacobian(stage);
    compiled.assemble_state_jacobian(stage);
    stage.update_projected_dynamics_residual();
    compiled.prepare_residual_problem(stage);
    compiled.assemble_residual(stage);

    REQUIRE(stage.nsp_.s_c_stacked.isApprox(A_default));
    REQUIRE(stage.nsp_.s_c_stacked_0_K.isApprox(B_default));
    REQUIRE(stage.nsp_.s_c_stacked_0_k.isApprox(a_default));
}

TEST_CASE("ordered equality blocks change factorization row order but preserve unique particular solve") {
    using namespace moto;
    using namespace moto::solver::projection;

    auto prob = make_partitioned_mixed_problem();

    ns_sqp::data stage_default(prob);
    solver::ns_riccati::generic_solver solver;
    stage_default.update_approximation(node_data::update_mode::eval_all);
    solver.ns_factorization(&stage_default);

    const matrix A_default = stage_default.nsp_.s_c_stacked;
    const matrix B_default = stage_default.nsp_.s_c_stacked_0_K;
    const vector a_default = stage_default.nsp_.s_c_stacked_0_k;
    const matrix u_y_K_default = stage_default.nsp_.u_y_K;
    const vector u_y_k_default = stage_default.nsp_.u_y_k;

    ns_sqp::data stage_reordered(prob);
    stage_reordered.default_elimination_stage_.configure_blocks(
        stage_reordered.ns,
        stage_reordered.nc,
        stage_reordered.nu,
        std::vector<equality_block_spec>{
            {.kind = equality_block_kind::state_input, .source_begin = 0, .source_count = 1, .group = "c0"},
            {.kind = equality_block_kind::projected_state, .source_begin = 0, .source_count = 1, .group = "s0"},
            {.kind = equality_block_kind::state_input, .source_begin = 1, .source_count = 1, .group = "c1"},
            {.kind = equality_block_kind::projected_state, .source_begin = 1, .source_count = 1, .group = "s1"},
        });
    stage_reordered.update_approximation(node_data::update_mode::eval_all);
    solver.ns_factorization(&stage_reordered);

    matrix A_expected(4, 2);
    A_expected.row(0) = A_default.row(2);
    A_expected.row(1) = A_default.row(0);
    A_expected.row(2) = A_default.row(3);
    A_expected.row(3) = A_default.row(1);

    matrix B_expected(4, 2);
    B_expected.row(0) = B_default.row(2);
    B_expected.row(1) = B_default.row(0);
    B_expected.row(2) = B_default.row(3);
    B_expected.row(3) = B_default.row(1);

    vector a_expected(4);
    a_expected(0) = a_default(2);
    a_expected(1) = a_default(0);
    a_expected(2) = a_default(3);
    a_expected(3) = a_default(1);

    REQUIRE(stage_reordered.nsp_.s_c_stacked.isApprox(A_expected));
    REQUIRE(stage_reordered.nsp_.s_c_stacked_0_K.isApprox(B_expected));
    REQUIRE(stage_reordered.nsp_.s_c_stacked_0_k.isApprox(a_expected));
    REQUIRE(stage_reordered.default_elimination_stage_.rows[0].group == "c0");
    REQUIRE(stage_reordered.default_elimination_stage_.rows[1].group == "s0");
    REQUIRE(stage_reordered.default_elimination_stage_.rows[2].group == "c1");
    REQUIRE(stage_reordered.default_elimination_stage_.rows[3].group == "s1");

    REQUIRE(stage_default.rank_status_ == solver::ns_riccati::rank_status::fully_constrained);
    REQUIRE(stage_reordered.rank_status_ == solver::ns_riccati::rank_status::fully_constrained);
    REQUIRE(stage_default.nsp_.Z_u.cols() == 0);
    REQUIRE(stage_reordered.nsp_.Z_u.cols() == 0);
    REQUIRE(stage_reordered.nsp_.u_y_K.isApprox(u_y_K_default));
    REQUIRE(stage_reordered.nsp_.u_y_k.isApprox(u_y_k_default));
}

TEST_CASE("ordered elimination preserves admissible subspace for partially constrained mixed blocks") {
    using namespace moto;
    using namespace moto::solver::projection;

    auto prob = make_partially_constrained_mixed_problem();

    ns_sqp::data stage_default(prob);
    solver::ns_riccati::generic_solver solver;
    stage_default.update_approximation(node_data::update_mode::eval_all);
    solver.ns_factorization(&stage_default);

    ns_sqp::data stage_reordered(prob);
    stage_reordered.default_elimination_stage_.configure_blocks(
        stage_reordered.ns,
        stage_reordered.nc,
        stage_reordered.nu,
        std::vector<equality_block_spec>{
            {.kind = equality_block_kind::state_input, .source_begin = 0, .source_count = 1, .group = "c"},
            {.kind = equality_block_kind::projected_state, .source_begin = 0, .source_count = 1, .group = "s"},
        });
    stage_reordered.update_approximation(node_data::update_mode::eval_all);
    solver.ns_factorization(&stage_reordered);

    REQUIRE(stage_default.rank_status_ == solver::ns_riccati::rank_status::constrained);
    REQUIRE(stage_reordered.rank_status_ == solver::ns_riccati::rank_status::constrained);
    REQUIRE(stage_default.nsp_.Z_u.cols() == 1);
    REQUIRE(stage_reordered.nsp_.Z_u.cols() == 1);

    const auto &A = stage_reordered.nsp_.s_c_stacked;
    const auto &B = stage_reordered.nsp_.s_c_stacked_0_K;
    const auto &a = stage_reordered.nsp_.s_c_stacked_0_k;
    REQUIRE((A * stage_reordered.nsp_.Z_u).isZero(1e-12));
    REQUIRE((A * stage_reordered.nsp_.u_y_K - B).isZero(1e-12));
    REQUIRE((A * stage_reordered.nsp_.u_y_k - a).isZero(1e-12));

    auto projector = [](const matrix &Z) {
        return Z * (Z.transpose() * Z).inverse() * Z.transpose();
    };
    REQUIRE(projector(stage_reordered.nsp_.Z_u).isApprox(projector(stage_default.nsp_.Z_u), 1e-12));
}

TEST_CASE("dense reference snapshot tracks reduced step and dual recovery") {
    using namespace moto;
    using namespace moto::solver::projection;

    ns_sqp::data stage(make_partially_constrained_problem());
    solver::ns_riccati::generic_solver solver;
    dense_snapshot snapshot;

    stage.update_approximation(node_data::update_mode::eval_all);
    solver.ns_factorization(&stage);
    solver.riccati_recursion(&stage, nullptr);
    solver.compute_primal_sensitivity(&stage);

    snapshot_dense_reduced_step(stage, snapshot);

    const auto &out = snapshot.out;

    REQUIRE(stage.rank_status_ == solver::ns_riccati::rank_status::constrained);
    REQUIRE(out.info.rank_status == rank_case::constrained);
    REQUIRE(out.has_factorization);
    REQUIRE(out.has_reduced_step);
    REQUIRE(!out.has_duals);
    REQUIRE(out.T_u.rows() == stage.nsp_.Z_u.rows());
    REQUIRE(out.T_u.cols() == stage.nsp_.Z_u.cols());
    REQUIRE(out.T_y.rows() == stage.nsp_.Z_y.rows());
    REQUIRE(out.T_y.cols() == stage.nsp_.Z_y.cols());
    REQUIRE(out.K_red.rows() == stage.nsp_.z_K.rows());
    REQUIRE(out.K_red.cols() == stage.nsp_.z_K.cols());
    REQUIRE(out.k_red.size() == stage.nsp_.z_k.size());
    REQUIRE(out.T_u.isApprox(stage.nsp_.Z_u));
    REQUIRE(out.T_y.isApprox(stage.nsp_.Z_y));
    REQUIRE(out.K_red.isApprox(stage.nsp_.z_K));
    REQUIRE(out.k_red.isApprox(stage.nsp_.z_k));

    solver.finalize_primal_step(&stage);
    solver.fwd_linear_rollout(&stage, nullptr);
    solver.finalize_dual_newton_step(&stage);

    snapshot_dense_duals(stage, snapshot);
    REQUIRE(snapshot.out.has_duals);
    REQUIRE(snapshot.out.dlbd_proj.size() == stage.d_lbd_s_c.size());
    REQUIRE(snapshot.out.dlbd_proj.isApprox(stage.d_lbd_s_c));
}
