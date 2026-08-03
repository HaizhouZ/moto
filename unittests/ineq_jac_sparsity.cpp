#include <catch2/catch_test_macros.hpp>

#include <cstdlib>

#include <moto/ocp/impl/func.hpp>
#include <moto/ocp/impl/node_data.hpp>
#include <moto/ocp/ineq_constr.hpp>
#include <moto/ocp/cost.hpp>

namespace {
const bool force_sync_codegen_for_test = []() {
    setenv("MOTO_SYNC_CODEGEN", "1", 1);
    return true;
}();

using namespace moto;

TEST_CASE("inequality jacobians preserve generated structure") {
    auto [x, y] = sym::states("x_sparse_jac", 3);
    (void)y;
    auto prob = stage_ocp::create();

    auto eye = ineq_constr::create(
        "eye_box",
        var_inarg_list(var_list{x}),
        static_cast<const cs::SX &>(x),
        vector::Constant(3, scalar_t(-1)),
        vector::Constant(3, scalar_t(1)),
        approx_order::first);
    auto diag = ineq_constr::create("diag_box", var_inarg_list(var_list{x}),
        scalar_t(2) * static_cast<const cs::SX &>(x),
        vector::Constant(3, scalar_t(-1)),
        vector::Constant(3, scalar_t(1)),
        approx_order::first);
    auto dense = ineq_constr::create(
        "dense_box",
        var_inarg_list(var_list{x}),
        cs::SX::vertcat({x(0) + x(1), x(1) + x(2), x(0) + x(2)}),
        vector::Constant(3, scalar_t(-1)),
        vector::Constant(3, scalar_t(1)),
        approx_order::first);

    prob->add(*eye);
    prob->add(*diag);
    prob->add(*dense);
    prob->wait_until_ready();

    auto &eye_func = dynamic_cast<const generic_func &>(*eye);
    auto &diag_func = dynamic_cast<const generic_func &>(*diag);
    auto &dense_func = dynamic_cast<const generic_func &>(*dense);

    REQUIRE(eye_func.jac_sparsity().size() == 1);
    REQUIRE(diag_func.jac_sparsity().size() == 1);
    REQUIRE(dense_func.jac_sparsity().size() == 1);
    REQUIRE(eye_func.jac_sparsity()[0].pattern == sparsity::eye);
  REQUIRE(diag_func.jac_sparsity()[0].pattern == sparsity::diag);
    REQUIRE(dense_func.jac_sparsity()[0].pattern == sparsity::dense);

    node_data data(prob);
    const auto &jac = data.dense().approx_[__ineq_x].jac_[__x];
  REQUIRE(jac.eye_panels_.size() == 1);
  REQUIRE(jac.diag_panels_.size() == 1);
  REQUIRE(jac.dense_panels_.size() == 1);
}

TEST_CASE("manual callbacks fall back to dense jacobians") {
    auto [x, y] = sym::states("x_manual_sparse_jac", 3);
    (void)y;
    auto prob = stage_ocp::create();

    auto eye = ineq_constr::create("manual_eye_box", approx_order::first, 3);
    auto &eye_func = dynamic_cast<generic_func &>(*eye);
    eye_func.add_argument(x);
    eye_func.set_jac_sparsity(x, sparsity::eye);
    eye->value = [](func_approx_data &d) { d.v_ = d[0]; };
    eye->jacobian = [](func_approx_data &d) { d.jac_[0].setOnes(); };

    auto diag = ineq_constr::create("manual_diag_box", approx_order::first, 3);
    auto &diag_func = dynamic_cast<generic_func &>(*diag);
    diag_func.add_argument(x);
    diag_func.set_jac_sparsity(x, sparsity::diag);
    diag->value = [](func_approx_data &d) { d.v_ = scalar_t(2) * d[0]; };
    diag->jacobian = [](func_approx_data &d) { d.jac_[0].setConstant(2); };

    prob->add(*eye);
    prob->add(*diag);
    prob->wait_until_ready();

    REQUIRE(dynamic_cast<const generic_func &>(*eye).jac_sparsity()[0].pattern == sparsity::dense);
    REQUIRE(dynamic_cast<const generic_func &>(*diag).jac_sparsity()[0].pattern == sparsity::dense);

    node_data data(prob);
    const auto &jac = data.dense().approx_[__ineq_x].jac_[__x];
    REQUIRE(jac.eye_panels_.empty());
    REQUIRE(jac.diag_panels_.empty());
    REQUIRE(jac.dense_panels_.size() == 2);
}

TEST_CASE("OCP finalize fuses structured blocks across callbacks") {
    auto [x0, y0] = sym::states("x_profile_0", 3);
    auto [x1, y1] = sym::states("x_profile_1", 3);
    (void)y0; (void)y1;
    auto prob = stage_ocp::create();
    const vector lb = vector::Constant(3, -1.);
    const vector ub = vector::Constant(3, 1.);
    auto c0 = ineq_constr::create("profile_box_0", var_inarg_list(var_list{x0}),
                                  static_cast<const cs::SX &>(x0), lb, ub,
                                  approx_order::first);
    auto c1 = ineq_constr::create("profile_box_1", var_inarg_list(var_list{x1}),
                                  static_cast<const cs::SX &>(x1), lb, ub,
                                  approx_order::first);
    auto q0 = generic_cost::from_vector(
        "profile_cost_0", var_inarg_list(var_list{x0}), x0);
    auto w1 = sym::params("profile_cost_1_weight_manual", 3,
                          vector::Constant(3, 2.));
    auto r1 = sym::params("profile_cost_1_reference_manual", 3,
                          vector::Zero(3));
    auto q1 = generic_cost::from_vector(
        "profile_cost_1", var_inarg_list(var_list{x1}), x1, w1, r1);
    REQUIRE(q0->weight()->default_value().isOnes());
    REQUIRE(q0->reference()->default_value().isZero());
    REQUIRE(q1->weight()->uid() == w1->uid());
    REQUIRE(q1->reference()->uid() == r1->uid());
    prob->add(*c0); prob->add(*c1); prob->add(*q0); prob->add(*q1);
    prob->wait_until_ready();
    node_data data(prob);
    const auto &jac = data.dense().approx_[__ineq_x].jac_[__x];
    REQUIRE(jac.eye_panels_.size() == 1);
    REQUIRE(jac.eye_panels_[0].rows_ == 6);
    const auto &hess = data.dense().lag_hess_[__x][__x];
    REQUIRE(hess.diag_panels_.size() == 2);
    REQUIRE(hess.diag_panels_[0].row_st_ == hess.diag_panels_[1].row_st_);
    REQUIRE(hess.diag_panels_[0].rows_ == hess.diag_panels_[1].rows_);
}
} // namespace
