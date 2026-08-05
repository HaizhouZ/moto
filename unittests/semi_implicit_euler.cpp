#include <catch2/catch_test_macros.hpp>

#include <moto/ocp/dynamics/dense_dynamics.hpp>
#include <moto/ocp/dynamics/semi_implicit_euler.hpp>
#include <moto/ocp/cost.hpp>
#include <moto/ocp/impl/node_data.hpp>
#include <moto/multibody/quaternion.hpp>
#include <moto/solver/ns_riccati/ns_riccati_data.hpp>
#include <moto/solver/ns_sqp.hpp>

#include <chrono>
#include <cstdint>

namespace moto {
namespace {

struct fixture {
  var q, qn, v, vn, u;
  dynamics semi, dense;
  ocp_ptr_t semi_problem, dense_problem;
  std::unique_ptr<node_data> semi_data, dense_data;

  fixture() {
    std::tie(q, qn) = sym::states("semi_test_q", 6);
    std::tie(v, vn) = sym::states("semi_test_v", 6);
    u = sym::inputs("semi_test_u", 4);
    cs::SX a = cs::SX::eye(6), b = -.02 * cs::SX::eye(6);
    a(cs::Slice(3, 6), cs::Slice(3, 6)) = cs::SX::reshape(
        cs::SX(std::vector<double>{1.1, -.1, .05, .2, .9, -.04,
                                   -.08, .12, 1.05}), 3, 3);
    b(cs::Slice(3, 6), cs::Slice(3, 6)) = cs::SX::reshape(
        cs::SX(std::vector<double>{-.02, .001, -.002, -.003, -.018, .001,
                                   .002, -.001, -.021}), 3, 3);
    cs::SX g = cs::SX::reshape(
        cs::SX(std::vector<double>{.1, .2, -.1, .05, -.2, .1,
                                   -.1, .3, .2, -.05, .15, -.2,
                                   .2, -.1, .1, .2, -.1, .05,
                                   -.05, .2, -.15, .1, .25, -.1}), 6, 4);
    const cs::SX residual = cs::SX::vertcat(
        {cs::SX::mtimes(a, qn) + cs::SX::mtimes(b, vn) - q,
         vn - v - cs::SX::mtimes(g, u)});
    semi = dynamics(new semi_implicit_euler(
        "semi_test_sparse", residual, semi_implicit_euler::state_t::pos_vel,
        approx_order::first));
    dense = dynamics(new dense_dynamics(
        "semi_test_dense", residual, approx_order::first));
    semi_problem = ocp::create();
    dense_problem = ocp::create();
    semi_problem->add(*semi);
    dense_problem->add(*dense);
    semi_problem->wait_until_ready();
    dense_problem->wait_until_ready();
    semi_data = std::make_unique<node_data>(semi_problem);
    dense_data = std::make_unique<node_data>(dense_problem);
    const vector xq = vector::LinSpaced(6, -.3, .4);
    const vector xv = vector::LinSpaced(6, .2, -.1);
    const vector yq = vector::LinSpaced(6, -.15, .35);
    const vector yv = vector::LinSpaced(6, .1, -.25);
    const vector xu = vector::LinSpaced(4, -.2, .3);
    for (node_data *data : {semi_data.get(), dense_data.get()}) {
      data->sym_val().get(q) = xq;
      data->sym_val().get(v) = xv;
      data->sym_val().get(qn) = yq;
      data->sym_val().get(vn) = yv;
      data->sym_val().get(u) = xu;
      data->update_approximation(node_data::update_mode::eval_all);
    }
    semi->compute_project_derivatives(semi_data->data(semi));
    dense->compute_project_derivatives(dense_data->data(dense));
  }
};

} // namespace

TEST_CASE("semi-implicit projections match dense dynamics") {
  fixture f;
  const auto &semi_approx = f.semi_data->data(f.semi).as<
      semi_implicit_euler::approx_data>();
  INFO("inverse error = " <<
       (f.semi_data->dense().approx_[__dyn].jac_[__y].dense() *
            semi_approx.inverse_.dense() -
        matrix::Identity(12, 12)).norm());
  INFO("Fy error = " <<
       (f.semi_data->dense().approx_[__dyn].jac_[__y].dense() -
        f.dense_data->dense().approx_[__dyn].jac_[__y].dense()).norm());
  INFO("Fx error = " << (f.semi_data->dense().proj_f_x().dense() -
                          f.dense_data->dense().proj_f_x().dense()).norm());
  INFO("semi PFx error = " <<
       (semi_approx.inverse_.dense() *
            f.semi_data->dense().approx_[__dyn].jac_[__x].dense() -
        f.semi_data->dense().proj_f_x().dense()).norm());
  INFO("dense PFx error = " <<
       (semi_approx.inverse_.dense() *
            f.dense_data->dense().approx_[__dyn].jac_[__x].dense() -
        f.dense_data->dense().proj_f_x().dense()).norm());
  INFO("dense raw/proj error = " <<
       (f.dense_data->dense().approx_[__dyn].jac_[__x].dense() -
        f.dense_data->dense().proj_f_x().dense()).norm());
  REQUIRE(f.semi_data->dense().proj_f_x().dense().isApprox(
      f.dense_data->dense().proj_f_x().dense(), 1e-12));
  REQUIRE(f.semi_data->dense().proj_f_u().dense().isApprox(
      f.dense_data->dense().proj_f_u().dense(), 1e-12));
  REQUIRE(f.semi_data->dense().proj_f_res().isApprox(
      f.dense_data->dense().proj_f_res(), 1e-12));

  vector rhs = vector::LinSpaced(12, -.4, .6), semi_out(12), dense_out(12);
  f.semi->apply_jac_y_inverse_transpose(f.semi_data->data(f.semi), rhs,
                                         semi_out);
  f.dense->apply_jac_y_inverse_transpose(f.dense_data->data(f.dense), rhs,
                                          dense_out);
  REQUIRE(semi_out.isApprox(dense_out, 1e-12));

  const auto &profiles =
      static_cast<const semi_implicit_euler &>(*f.semi).projected_profiles();
  REQUIRE(profiles.size() == 3);
  REQUIRE(profiles[0].rows == 12);
  REQUIRE(profiles[0].nnz() < profiles[0].rows * profiles[0].cols);
  const auto stored_nnz = [](const sparse_matrix &value) {
    size_t count = 0;
    for (const auto &panel : value.dense_panels_)
      count += panel.rows_ * panel.cols_;
    if (value.diagonal_segments_.empty()) {
      for (const auto &panel : value.diag_panels_)
        count += panel.rows_;
    } else {
      for (const auto &segment : value.diagonal_segments_)
        count += segment.rows;
    }
    for (const auto &panel : value.eye_panels_)
      count += panel.rows_;
    return count;
  };
  REQUIRE(stored_nnz(f.semi_data->dense().proj_f_x()) ==
          profiles[0].nnz() + profiles[1].nnz());
  REQUIRE(stored_nnz(f.semi_data->dense().proj_f_u()) == profiles[2].nnz());
  for (const auto *projected : {&f.semi_data->dense().proj_f_x(),
                                &f.semi_data->dense().proj_f_u()}) {
    if (!projected->diagonal_segments_.empty()) {
      REQUIRE(projected->diag_panels_.size() == 1);
      for (const auto &segment : projected->diagonal_segments_) {
        const auto *pointer =
            projected->diag_panels_[segment.storage_panel].data_.data() +
            segment.storage_offset;
        REQUIRE(reinterpret_cast<std::uintptr_t>(pointer) %
                    EIGEN_MAX_ALIGN_BYTES ==
                0);
      }
    }
  }

  if (std::getenv("MOTO_BENCH_SEMI_IMPLICIT")) {
    constexpr size_t warmup = 1000, runs = 20000;
    const auto measure = [&](node_data &data, const dynamics &dyn) {
      for (size_t i = 0; i < warmup; ++i) {
        data.update_approximation(node_data::update_mode::eval_all);
        dyn->compute_project_derivatives(data.data(dyn));
      }
      const auto start = std::chrono::steady_clock::now();
      for (size_t i = 0; i < runs; ++i) {
        data.update_approximation(node_data::update_mode::eval_all);
        dyn->compute_project_derivatives(data.data(dyn));
      }
      return std::chrono::duration<double, std::nano>(
                 std::chrono::steady_clock::now() - start)
                 .count() /
             runs;
    };
    const double semi_ns = measure(*f.semi_data, f.semi);
    const double dense_ns = measure(*f.dense_data, f.dense);
    fmt::println("semi_implicit={} ns dense_fallback={} ns speedup={}x",
                 semi_ns, dense_ns, dense_ns / semi_ns);
    for (size_t i = 0; i < profiles.size(); ++i)
      fmt::println("PF[{}]: {}x{}, nnz={}, blocks={}", i,
                   profiles[i].rows, profiles[i].cols, profiles[i].nnz(),
                   profiles[i].row_blocks.empty()
                       ? 0
                       : profiles[i].row_blocks.size() - 1);
  }
}

TEST_CASE("position Euler is a complete kinematic dynamics") {
  auto [q, qn] = multibody::quaternion::create("position_euler_q");
  auto velocity = sym::inputs("position_euler_velocity", 3);
  const cs::SX integrated = q->symbolic_integrate(
      *q, .1 * static_cast<const cs::SX &>(*velocity));
  const cs::SX residual = q->symbolic_difference(*qn, integrated);
  dynamics euler(new semi_implicit_euler(
      "position_euler", residual, semi_implicit_euler::state_t::pos,
      approx_order::first));
  dynamics fallback(new dense_dynamics("position_euler_dense", residual,
                                       approx_order::first));
  auto euler_problem = ocp::create(), dense_problem = ocp::create();
  euler_problem->add(*euler);
  dense_problem->add(*fallback);
  euler_problem->wait_until_ready();
  dense_problem->wait_until_ready();
  node_data euler_data(euler_problem), dense_data(dense_problem);
  for (node_data *data : {&euler_data, &dense_data}) {
    data->sym_val().get(*q) = multibody::quaternion::identity();
    data->sym_val().get(*qn) = multibody::quaternion::identity();
    data->sym_val().get(velocity) = vector::LinSpaced(3, -.1, .2);
    data->update_approximation(node_data::update_mode::eval_all);
  }
  euler->compute_project_derivatives(euler_data.data(euler));
  fallback->compute_project_derivatives(dense_data.data(fallback));
  const auto &euler_approx = euler_data.data(euler).as<
      semi_implicit_euler::approx_data>();
  INFO("position inverse error = " <<
       (euler_data.dense().approx_[__dyn].jac_[__y].dense() *
            euler_approx.inverse_.dense() -
        matrix::Identity(3, 3)).norm());
  INFO("position PFx error = " <<
       (euler_data.dense().proj_f_x().dense() -
        dense_data.dense().proj_f_x().dense()).norm());
  REQUIRE(euler_data.dense().proj_f_x().dense().isApprox(
      dense_data.dense().proj_f_x().dense(), 1e-12));
  REQUIRE(euler_data.dense().proj_f_u().dense().isApprox(
      dense_data.dense().proj_f_u().dense(), 1e-12));
  REQUIRE(euler_data.dense().proj_f_res().isApprox(
      dense_data.dense().proj_f_res(), 1e-12));
  vector rhs = vector::LinSpaced(3, -.3, .4), euler_out(3), dense_out(3);
  euler->apply_jac_y_inverse_transpose(euler_data.data(euler), rhs, euler_out);
  fallback->apply_jac_y_inverse_transpose(
      dense_data.data(fallback), rhs, dense_out);
  REQUIRE(euler_out.isApprox(dense_out, 1e-12));
}

TEST_CASE("multiple dynamics use independent local projections and shared input") {
  auto [x1, y1] = sym::states("multi_projection_x1");
  auto [x2, y2] = sym::states("multi_projection_x2");
  auto [x3, y3] = sym::states("multi_projection_x3");
  auto shared_u = sym::inputs("multi_projection_shared_u");
  auto dense_u = sym::inputs("multi_projection_dense_u");
  auto euler_u = sym::inputs("multi_projection_euler_u");
  const cs::SX &sx1 = x1, &sy1 = y1, &sx2 = x2, &sy2 = y2;
  const cs::SX &sx3 = x3, &sy3 = y3, &ssu = shared_u;
  const cs::SX &sdu = dense_u, &seu = euler_u;
  dynamics dense(new dense_dynamics(
      "multi_projection_dense", 2 * sy1 - sx1 - sdu - ssu,
      approx_order::first));
  dynamics euler(new semi_implicit_euler(
      "multi_projection_euler", 3 * sy2 - 2 * sx2 - 2 * seu - 4 * ssu,
      semi_implicit_euler::state_t::pos, approx_order::first));
  dynamics autonomous(new dense_dynamics(
      "multi_projection_autonomous", 4 * sy3 - sx3, approx_order::first));
  dense->mark_shared_inputs({shared_u});
  euler->mark_shared_inputs({shared_u});
  auto ordering_cost = generic_cost::from_scalar(
      "multi_projection_ordering_cost", var_inarg_list{},
      seu * seu + ssu * ssu + sdu * sdu);

  auto problem = ocp::create();
  problem->add(*ordering_cost);
  problem->add(*dense);
  problem->add(*euler);
  problem->add(*autonomous);
  problem->wait_until_ready();
  node_data data(problem);
  data.sym_val().get(x1)(0) = 1.;
  data.sym_val().get(y1)(0) = 2.;
  data.sym_val().get(x2)(0) = 3.;
  data.sym_val().get(y2)(0) = 4.;
  data.sym_val().get(x3)(0) = 5.;
  data.sym_val().get(y3)(0) = 6.;
  data.sym_val().get(shared_u)(0) = .25;
  data.sym_val().get(dense_u)(0) = .5;
  data.sym_val().get(euler_u)(0) = .75;
  data.update_approximation(node_data::update_mode::eval_all);

  solver::ns_riccati::ns_riccati_data projected(&data);
  projected.update_projected_dynamics();
  projected.update_projected_dynamics_residual();
  matrix expected_fx = matrix::Zero(3, 3);
  expected_fx.diagonal() << -.5, -2. / 3., -.25;
  matrix expected_fu = matrix::Zero(3, 3);
  expected_fu.row(0) << -.5, 0., -.5;
  expected_fu.row(1) << 0., -2. / 3., -4. / 3.;
  vector expected_res(3);
  expected_res << 1.125, 7. / 6., 4.75;
  REQUIRE(data.dense().proj_f_x().dense().isApprox(expected_fx, 1e-12));
  REQUIRE(data.dense().proj_f_u().dense().isApprox(expected_fu, 1e-12));
  REQUIRE(data.dense().proj_f_res().isApprox(expected_res, 1e-12));

  vector rhs(3), result(3), expected_dual(3);
  rhs << 2., 3., 4.;
  expected_dual << 1., 1., 1.;
  projected.apply_jac_y_inverse_transpose(rhs, result);
  REQUIRE(result.isApprox(expected_dual, 1e-12));
}

TEST_CASE("multiple dynamics support regrouped phases and optimized initial state") {
  auto [x1, y1] = sym::states("multi_phase_x1");
  auto [x2, y2] = sym::states("multi_phase_x2");
  auto u1 = sym::inputs("multi_phase_u1");
  auto u2 = sym::inputs("multi_phase_u2");
  const cs::SX &sx1 = x1, &sy1 = y1, &sx2 = x2, &sy2 = y2;
  const cs::SX &su1 = u1, &su2 = u2;
  dynamics combined(new dense_dynamics(
      "multi_phase_combined",
      cs::SX::vertcat({sy1 - sx1 - su1, sy2 - sx2 - su2}),
      approx_order::first));
  dynamics split1(new dense_dynamics(
      "multi_phase_split1", sy1 - sx1 - su1, approx_order::first));
  dynamics split2(new dense_dynamics(
      "multi_phase_split2", sy2 - sx2 - su2, approx_order::first));
  auto running = generic_cost::from_scalar(
      "multi_phase_cost", var_inarg_list{},
      sx1 * sx1 + sx2 * sx2 + su1 * su1 + su2 * su2);

  auto first = stage_ocp::create(), second = stage_ocp::create();
  first->add(*combined);
  first->add(*running);
  second->add(*split1);
  second->add(*split2);
  second->add(*running);
  ns_sqp sqp(1);
  sqp.settings.initial_state = ns_sqp::initial_state_mode::optimized;
  sqp.add_stage(first, 1);
  sqp.add_stage(second, 2);
  auto &nodes = sqp.solver_nodes();
  for (auto *node : nodes) {
    node->sym_val().get(x1)(0) = 1.;
    node->sym_val().get(y1)(0) = 1.;
    node->sym_val().get(x2)(0) = 2.;
    node->sym_val().get(y2)(0) = 2.;
  }
  const auto result = sqp.update(20, false);
  REQUIRE(result.iter.result == ns_sqp::iter_result_t::success);
}

} // namespace moto
