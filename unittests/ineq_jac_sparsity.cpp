#include <catch2/catch_test_macros.hpp>

#include <cstdlib>

#include <moto/ocp/impl/func.hpp>
#include <moto/ocp/impl/node_data.hpp>
#include <moto/ocp/ineq_constr.hpp>

namespace {
const bool force_sync_codegen_for_test = []() {
    setenv("MOTO_SYNC_CODEGEN", "1", 1);
    return true;
}();

using namespace moto;

TEST_CASE("inequality jacobians are stored as dense panels") {
    auto [x, y] = sym::states("x_sparse_jac", 3);
    (void)y;
    auto prob = node_ocp::create();

    auto eye = ineq_constr::create(
        "eye_box",
        var_inarg_list(var_list{x}),
        static_cast<const cs::SX &>(x),
        vector::Constant(3, scalar_t(-1)),
        vector::Constant(3, scalar_t(1)),
        approx_order::first);
    auto diag = ineq_constr::create(
        "diag_box",
        var_inarg_list(var_list{x}),
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
    REQUIRE(eye_func.jac_sparsity()[0].pattern == sparsity::dense);
    REQUIRE(diag_func.jac_sparsity()[0].pattern == sparsity::dense);
    REQUIRE(dense_func.jac_sparsity()[0].pattern == sparsity::dense);

    node_data data(prob);
    const auto &jac = data.dense().approx_[__ineq_x].jac_[__x];
    REQUIRE(jac.eye_panels_.empty());
    REQUIRE(jac.diag_panels_.empty());
    REQUIRE(jac.dense_panels_.size() == 3);
}

TEST_CASE("manual jacobian sparsity declarations are ignored") {
    auto [x, y] = sym::states("x_manual_sparse_jac", 3);
    (void)y;
    auto prob = node_ocp::create();

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
} // namespace
