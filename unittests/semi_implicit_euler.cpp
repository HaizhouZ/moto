#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <moto/ocp/dynamics/dense_dynamics.hpp>
#include <moto/ocp/dynamics/semi_implicit_euler.hpp>
#include <moto/ocp/cost.hpp>
#include <moto/ocp/impl/node_data.hpp>
#include <moto/multibody/casadi_manifold.hpp>
#include <moto/solver/ns_riccati/ns_riccati_data.hpp>
#include <moto/solver/ns_sqp.hpp>

#include <chrono>
#include <cstdint>

namespace moto {
namespace {

class mock_lifted final : public generic_lifted {
public:
  using generic_lifted::generic_lifted;
  void compute_project_jacobians(func_approx_data &) const override {}
  void compute_project_residual(func_approx_data &) const override {}
  void apply_lifted_jacobian_inverse_transpose(
      func_approx_data &, vector_ref v, vector_ref dst) const override {
    dst = v;
  }

protected:
  clone_ptr clone() const override { return new mock_lifted(*this); }
};

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

TEST_CASE("generic lifted groups declare arbitrary primal lifted arguments") {
  auto lifted_input = sym::lifted("generic_lifted_l", 2);
  auto free_input = sym::inputs("generic_lifted_u", 1);
  const cs::SX &sl = lifted_input, &su = free_input;
  const cs::SX residual = sl + cs::SX::vertcat({su, 2. * su});
  auto group = std::make_shared<mock_lifted>(
      "generic_lifted_group", residual, approx_order::first, __lift);
  group->mark_lifted({lifted_input});

  auto problem = ocp::create();
  problem->add(*group);
  problem->wait_until_ready();

  REQUIRE(group->lifted_tdim() == 2);
  REQUIRE(group->lifted_args().size() == 1);
  REQUIRE(group->is_lifted(lifted_input));
  REQUIRE_FALSE(group->is_lifted(free_input));

  auto remapped_lifted = sym::lifted("generic_lifted_l_remapped", 2);
  auto remapped_handle = group->remap_arguments(
      {{lifted_input, remapped_lifted}});
  const auto *remapped = dynamic_cast<const mock_lifted *>(
      remapped_handle.get());
  REQUIRE(remapped != nullptr);
  REQUIRE(remapped->is_lifted(remapped_lifted));
  REQUIRE_FALSE(remapped->is_lifted(lifted_input));

  auto invalid_lifted = sym::lifted("generic_lifted_invalid", 1);
  const cs::SX &si = invalid_lifted;
  auto invalid = std::make_shared<mock_lifted>(
      "generic_lifted_non_square", cs::SX::vertcat({si, si}),
      approx_order::first, __lift);
  invalid->mark_lifted({invalid_lifted});
  auto invalid_problem = ocp::create();
  REQUIRE_THROWS_WITH(
      invalid_problem->add(*invalid),
      Catch::Matchers::ContainsSubstring("requires a square lifted Jacobian"));
}

TEST_CASE("lifted MX elimination graph preserves zero blocks and parameters") {
  auto [x, y] = sym::states("lifted_graph_x", 2);
  auto u = sym::inputs("lifted_graph_u", 2);
  auto l = sym::lifted("lifted_graph_l", 2);
  var regularization;
  const cs::SX &sx = x, &sy = y, &su = u, &sl = l;
  const cs::SX dynamics_equation = sy - sx - sl;
  const cs::SX group_equation = sy - su;

  dynamics dyn = std::make_shared<semi_implicit_euler>(
      "lifted_graph_dynamics", dynamics_equation,
      semi_implicit_euler::state_t::pos, approx_order::first);
  const dynamics source = dyn;
  auto group = std::make_shared<implicit_lifted>(
      "lifted_graph_constraint", group_equation,
      var_inarg_list{l}, approx_order::first);

  bool saw_shaped_zero = false;
  dyn = dyn->with_elimination_graph(
      [&](const lifted_symbolic_system &system) {
        const auto dyn_l = system.jac(*source, *l);
        const auto dyn_l_again = system.jac(*source, *l);
        const auto dyn_y = system.jac(*source, *y);
        const auto lift_y = system.jac(*group, *y);
        const auto lift_l = system.jac(*group, *l);
        saw_shaped_zero = system.lift_l.size1() == 2 &&
                          system.lift_l.size2() == 2 &&
                          system.lift_l.nnz() == 0 &&
                          lift_l.value.nnz() == 0 &&
                          dyn_l_again.value.sparsity() ==
                              dyn_l.value.sparsity() &&
                          system.residual(*group).size1() ==
                              2;
        regularization = lift_l.param(1e-4);
        const cs::MX regularized_lift_l =
            lift_l.add_diag(*regularization);
        const cs::MX h_l = cs::MX::vertcat(std::vector<cs::MX>{
            cs::MX::horzcat(
                std::vector<cs::MX>{dyn_y.value, dyn_l.value}),
            cs::MX::horzcat(
                std::vector<cs::MX>{lift_y.value, regularized_lift_l})});
        const auto factor = system.solve(h_l);
        const auto solve = [&](const cs::MX &rhs) {
          return factor.solve(rhs);
        };
        return system.eliminate(
            solve, {{"regularized_lift_l", regularized_lift_l}});
      }, {group});

  REQUIRE(dyn.get() != source.get());
  REQUIRE_FALSE(source->has_elimination_graph());
  REQUIRE(source->subconstraints().empty());
  REQUIRE(dyn->has_elimination_graph());

  auto problem = ocp::create();
  problem->add(*dyn);
  problem->wait_until_ready();

  REQUIRE(problem->contains(*group));
  REQUIRE(dyn->subconstraints().size() == 1);
  REQUIRE(dyn->owns_subconstraint(*group));
  REQUIRE(saw_shaped_zero);
  REQUIRE(regularization);
  REQUIRE(regularization->name() ==
          "lifted_graph_constraint_lifted_graph_l_regularization");
  REQUIRE(regularization->field() == __p);
  REQUIRE(problem->contains(*regularization));
  REQUIRE_FALSE(dyn->has_arg(*regularization));
  REQUIRE(dyn->elimination_parameters().size() == 1);
  const auto &profile = problem->linear_profile();
  const auto panel_binding = std::ranges::find_if(
      profile.lifted_program->input_bindings,
      [&](const lifted_graph_input_binding &binding) {
          return binding.source ==
                     lifted_graph_input_binding::kind::jacobian_panel &&
                 binding.equation == __dyn && binding.variable == __l;
      });
  REQUIRE(panel_binding != profile.lifted_program->input_bindings.end());
  REQUIRE(std::ranges::count_if(
              profile.lifted_program->input_bindings,
              [&](const lifted_graph_input_binding &binding) {
                  return binding.source ==
                             lifted_graph_input_binding::kind::jacobian_panel &&
                         binding.equation == __dyn &&
                         binding.variable == __l;
              }) == 1);
  const size_t panel_index = static_cast<size_t>(std::distance(
      profile.lifted_program->input_bindings.begin(), panel_binding));
  REQUIRE_FALSE(
      profile.lifted_program->input_layouts[panel_index].panels.empty());
  REQUIRE_FALSE(profile.get(linear_target::lifted_projection, __y, __u)
                    .empty());
  REQUIRE_FALSE(profile.get(linear_target::lifted_projection, __l, __x)
                    .empty());
  REQUIRE(profile.lifted_intermediates.size() == 1);
  const auto &intermediate = profile.lifted_intermediates.front();
  REQUIRE(intermediate.name == "regularized_lift_l");
  REQUIRE(intermediate.rows == 2);
  REQUIRE(intermediate.cols == 2);
  REQUIRE(intermediate.layout.panels.size() == 1);
  REQUIRE(intermediate.layout.panels.front().pattern == sparsity::diag);
  node_data runtime(problem);
  REQUIRE(runtime.sym_val().get(regularization)(0) == 1e-4);
  REQUIRE(runtime.data(dynamics(dyn))[*regularization](0) == 1e-4);
  REQUIRE(runtime.data(lifted(group))[*regularization](0) == 1e-4);
  runtime.sym_val().get(x) << 0.2, -0.3;
  runtime.sym_val().get(y) << 0.5, 0.7;
  runtime.sym_val().get(u) << -0.4, 0.6;
  runtime.sym_val().get(l) << 0.1, -0.2;
  runtime.update_approximation(node_data::update_mode::eval_all);
  auto &dyn_data = runtime.data(dynamics(dyn));
  dyn->compute_project_derivatives(dyn_data);
  auto &projected = dyn_data.as<generic_dynamics::approx_data>();
  matrix pivot(4, 4), rhs(4, 5);
  pivot << runtime.dense().approx_[__dyn].jac_[__y].dense(),
      runtime.dense().approx_[__dyn].jac_[__l].dense(),
      runtime.dense().approx_[__lift].jac_[__y].dense(),
      runtime.dense().approx_[__lift].jac_[__l].dense() +
          1e-4 * matrix::Identity(2, 2);
  rhs << runtime.dense().approx_[__dyn].jac_[__x].dense(),
      runtime.dense().approx_[__dyn].jac_[__u].dense(),
      runtime.dense().approx_[__dyn].v_,
      runtime.dense().approx_[__lift].jac_[__x].dense(),
      runtime.dense().approx_[__lift].jac_[__u].dense(),
      runtime.dense().approx_[__lift].v_;
  const matrix expected = pivot.fullPivLu().solve(rhs);
  REQUIRE(runtime.dense().proj_f_x().dense().isApprox(
      expected.topLeftCorner(2, 2), 1e-9));
  REQUIRE(runtime.dense().proj_f_u().dense().isApprox(
      expected.block(0, 2, 2, 2), 1e-9));
  REQUIRE(runtime.dense().proj_f_res().isApprox(expected.block(0, 4, 2, 1),
                                                1e-9));
  REQUIRE(projected.proj_l_x_.dense().isApprox(
      expected.bottomLeftCorner(2, 2), 1e-9));
  REQUIRE(projected.proj_l_u_.dense().isApprox(
      expected.block(2, 2, 2, 2), 1e-9));
  REQUIRE(projected.proj_l_res_.isApprox(expected.block(2, 4, 2, 1),
                                         1e-10));
  matrix action_rhs = matrix::Random(4, 2), action, transpose;
  dyn->solve_stage_lifted_system(dyn_data, action_rhs, action, false);
  dyn->solve_stage_lifted_system(dyn_data, action_rhs, transpose, true);
  REQUIRE(action.isApprox(pivot.fullPivLu().solve(action_rhs), 1e-10));
  REQUIRE(transpose.isApprox(
      pivot.transpose().fullPivLu().solve(action_rhs), 1e-10));

  runtime.sym_val().get(regularization)(0) = 2e-2;
  runtime.update_approximation(node_data::update_mode::eval_derivatives);
  dyn->compute_project_derivatives(dyn_data);
  pivot.bottomRightCorner(2, 2) =
      runtime.dense().approx_[__lift].jac_[__l].dense() +
      2e-2 * matrix::Identity(2, 2);
  rhs.leftCols(4) << runtime.dense().approx_[__dyn].jac_[__x].dense(),
      runtime.dense().approx_[__dyn].jac_[__u].dense(),
      runtime.dense().approx_[__lift].jac_[__x].dense(),
      runtime.dense().approx_[__lift].jac_[__u].dense();
  const matrix refreshed = pivot.fullPivLu().solve(rhs.leftCols(4));
  REQUIRE(runtime.dense().proj_f_x().dense().isApprox(
      refreshed.topLeftCorner(2, 2), 1e-9));
  REQUIRE(runtime.dense().proj_f_u().dense().isApprox(
      refreshed.block(0, 2, 2, 2), 1e-9));
}

TEST_CASE("lifted graph artifact identity includes elimination algebra") {
  auto [x, y] = sym::states("lifted_identity_x", 2);
  auto u = sym::inputs("lifted_identity_u", 2);
  auto l = sym::lifted("lifted_identity_l", 2);
  const cs::SX &sx = x, &sy = y, &su = u, &sl = l;
  dynamics source = std::make_shared<semi_implicit_euler>(
      "lifted_identity_dynamics", sy - sx - sl,
      semi_implicit_euler::state_t::pos, approx_order::first);
  auto constraint = std::make_shared<implicit_lifted>(
      "lifted_identity_constraint", sy - su, var_inarg_list{l},
      approx_order::first);
  const auto make_problem = [&](scalar_t response_scale) {
    dynamics generated = source->with_elimination_graph(
        [response_scale](const lifted_symbolic_system &system) {
          const auto factor = system.solve(system.h_l());
          const auto solve = [&](const cs::MX &rhs) {
            return response_scale * factor.solve(rhs);
          };
          return system.eliminate(solve);
        }, {constraint});
    auto problem = ocp::create();
    problem->add(*generated);
    problem->wait_until_ready();
    return problem;
  };

  const auto first = make_problem(1.);
  const auto equivalent = make_problem(1.);
  const auto different = make_problem(2.);
  const auto &first_identity =
      first->linear_profile().lifted_program->artifact_identity;
  REQUIRE_FALSE(first_identity.empty());
  REQUIRE(equivalent->linear_profile().lifted_program->artifact_identity ==
          first_identity);
  REQUIRE(different->linear_profile().lifted_program->artifact_identity !=
          first_identity);
}

TEST_CASE("integrated lifted presolve retains overlapping state Hessians") {
  auto [x, y] = sym::states("lifted_presolve_x");
  auto u = sym::inputs("lifted_presolve_u");
  auto l = sym::lifted("lifted_presolve_l");
  const cs::SX &sx = x, &sy = y, &su = u, &sl = l;
  dynamics dyn = std::make_shared<semi_implicit_euler>(
      "lifted_presolve_dyn", sy - sx - sl,
      semi_implicit_euler::state_t::pos, approx_order::first);
  auto lifting = std::make_shared<implicit_lifted>(
      "lifted_presolve_constraint", sl - su, var_inarg_list{l},
      approx_order::first);
  dyn = dyn->with_elimination_graph(
      [](const lifted_symbolic_system &system) {
        const auto factor = system.solve(system.h_l());
        return system.eliminate(
            [&](const cs::MX &rhs) { return factor.solve(rhs); });
      },
      {lifting});
  auto state_0 = generic_cost::from_scalar(
      "lifted_presolve_state_0", var_inarg_list{},
      3. * (sx - 1.) * (sx - 1.));
  auto state_1 = generic_cost::from_scalar(
      "lifted_presolve_state_1", var_inarg_list{},
      5. * (sx + .25) * (sx + .25));
  auto problem = ocp::create();
  problem->add(*dyn);
  problem->add(*state_0);
  problem->add(*state_1);
  problem->wait_until_ready();

  node_data runtime(problem);
  runtime.sym_val().get(x).setZero();
  runtime.sym_val().get(y).setZero();
  runtime.sym_val().get(u).setZero();
  runtime.sym_val().get(l).setZero();
  runtime.prepare_linear_plan();
  runtime.update_approximation(node_data::update_mode::eval_all);
  solver::ns_riccati::ns_riccati_data projected(&runtime);
  projected.prepare_linear_backend();
  solver::ns_riccati::generic_solver solver;
  std::array<solver::ns_riccati::ns_riccati_data *, 1> stages{&projected};
  solver.prepare_ocp_linear_graph(stages);
  setenv("MOTO_VERIFY_NSP_GRAPH", "1", 1);
  solver.ns_factorization(&projected);
  unsetenv("MOTO_VERIFY_NSP_GRAPH");

  matrix expected = matrix::Zero(projected.nx, projected.nx);
  linear_backend::write_dense(projected.Q_xx, expected);
  linear_backend::write_dense(projected.Q_xx_mod, expected);
  REQUIRE(projected.V_xx.isApprox(expected, 1e-12));
}

TEST_CASE("semi-implicit projections match dense dynamics") {
  fixture f;
  for (const dynamics &dyn : {f.semi, f.dense}) {
    const auto *group = dynamic_cast<const generic_lifted *>(dyn.get());
    REQUIRE(group != nullptr);
    REQUIRE(group->lifted_tdim() == 12);
    REQUIRE(group->lifted_args().size() == 2);
    for (const sym &arg : group->lifted_args()) {
      REQUIRE(arg.field() == __y);
      REQUIRE(group->is_lifted(arg));
    }
  }
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
  REQUIRE(stored_nnz(f.semi_data->dense().proj_f_x()) < 12 * 24);
  REQUIRE(stored_nnz(f.semi_data->dense().proj_f_u()) < 12 * 12);
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
  }
}

TEST_CASE("position Euler supports generic nonlinear manifolds") {
  const cs::SX base = cs::SX::sym("position_euler_base", 2);
  const cs::SX step = cs::SX::sym("position_euler_step");
  const cs::SX other = cs::SX::sym("position_euler_other", 2);
  const cs::SX rotated = cs::SX::vertcat(std::vector<cs::SX>{
      cs::SX::cos(step) * base(0) - cs::SX::sin(step) * base(1),
      cs::SX::sin(step) * base(0) + cs::SX::cos(step) * base(1)});
  const cs::SX difference = cs::SX::atan2(
      base(0) * other(1) - base(1) * other(0),
      base(0) * other(0) + base(1) * other(1));
  const vector identity = (vector(2) << 1., 0.).finished();
  auto [q, qn] = multibody::casadi_manifold::create(
      "position_euler_q", base, step, rotated, other, difference, identity);
  auto velocity = sym::inputs("position_euler_velocity");
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
    data->sym_val().get(*q) = identity;
    data->sym_val().get(*qn) = identity;
    data->sym_val().get(velocity)(0) = .2;
    data->update_approximation(node_data::update_mode::eval_all);
  }
  euler->compute_project_derivatives(euler_data.data(euler));
  fallback->compute_project_derivatives(dense_data.data(fallback));
  const auto &euler_approx = euler_data.data(euler).as<
      semi_implicit_euler::approx_data>();
  INFO("position inverse error = " <<
       (euler_data.dense().approx_[__dyn].jac_[__y].dense() *
            euler_approx.inverse_.dense() -
        matrix::Identity(1, 1)).norm());
  INFO("position PFx error = " <<
       (euler_data.dense().proj_f_x().dense() -
        dense_data.dense().proj_f_x().dense()).norm());
  REQUIRE(euler_data.dense().proj_f_x().dense().isApprox(
      dense_data.dense().proj_f_x().dense(), 1e-12));
  REQUIRE(euler_data.dense().proj_f_u().dense().isApprox(
      dense_data.dense().proj_f_u().dense(), 1e-12));
  REQUIRE(euler_data.dense().proj_f_res().isApprox(
      dense_data.dense().proj_f_res(), 1e-12));
  vector rhs(1), euler_out(1), dense_out(1);
  rhs(0) = -.3;
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
  sqp.stages().push_back(first->copy());
  sqp.stages().push_back(second->copy());
  sqp.stages().push_back(second->copy());
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
