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

class tracking_manifold final : public sym {
  public:
    inline static size_t difference_calls = 0;
    explicit tracking_manifold(const std::string &name) : sym(name, 2, __u) {
        tdim_ = 1;
    }
    cs::SX symbolic_difference(const cs::SX &x1,
                               const cs::SX &x0) const override {
        ++difference_calls;
        return (x1(0) - x0(0)) + scalar_t(2) * (x1(1) - x0(1));
    }

  protected:
    clone_ptr clone() const override { return new tracking_manifold(*this); }
};

TEST_CASE("CasADi symvar inference restores original symbol handles") {
    auto [x, y] = sym::states("auto_args_x", 2);
    auto u = sym::inputs("auto_args_u", 1);
    auto p = sym::params("auto_args_p", 1);
    auto f = std::make_shared<generic_constr>(
        "auto_args_func", cs::SX::vertcat({y - x, u + p}),
        approx_order::first);

    REQUIRE(f->in_args().size() == 4);
    const std::set<size_t> inferred{
        f->in_args()[0]->uid(), f->in_args()[1]->uid(),
        f->in_args()[2]->uid(), f->in_args()[3]->uid()};
    REQUIRE(inferred == std::set<size_t>{x->uid(), y->uid(), u->uid(), p->uid()});
    const var_list originals{x, y, u, p};
    for (const var &arg : f->in_args()) {
        const auto original = std::ranges::find_if(
            originals, [&](const var &candidate) {
                return candidate->uid() == arg->uid();
            });
        REQUIRE(original != originals.end());
        REQUIRE(arg.get() == original->get());
    }
}

TEST_CASE("tracking cost uses the value symbol manifold difference") {
    var q(new tracking_manifold("tracking_manifold_q"));
    global_registry::add(q);
    tracking_manifold::difference_calls = 0;

    auto cost = generic_cost::from_vector(
        "tracking_manifold_cost", var_inarg_list{}, q);

    REQUIRE(tracking_manifold::difference_calls == 1);
    REQUIRE(cost->reference()->dim() == q->dim());
    REQUIRE(cost->weight()->dim() == q->tdim());
}

TEST_CASE("codegen splits mixed dense and diagonal cost Hessians") {
    auto [x, y] = sym::states("mixed_hessian_x", 8);
    (void)y;
    const cs::SX residual = cs::SX::vertcat({
        x(0), x(1), x(2) * x(3), x(3) * x(4), x(2) * x(4),
        x(5), x(6), x(7)});
    auto cost = generic_cost::from_vector(
        "mixed_hessian_cost", var_inarg_list{}, residual);
    auto prob = stage_ocp::create();
    prob->add(*cost);
    prob->wait_until_ready();

    const auto &panels = cost->hess_panel_sparsity();
    REQUIRE(panels.size() == 3);
    REQUIRE(panels[0].block.pattern == sparsity::diag);
    REQUIRE(panels[0].block.row_offset == 0);
    REQUIRE(panels[0].block.rows == 2);
    REQUIRE(panels[1].block.pattern == sparsity::dense);
    REQUIRE(panels[1].block.row_offset == 2);
    REQUIRE(panels[1].block.rows == 3);
    REQUIRE(panels[2].block.pattern == sparsity::diag);
    REQUIRE(panels[2].block.row_offset == 5);
    REQUIRE(panels[2].block.rows == 3);

    node_data data(prob);
    data.sym_val().value_[__x] = vector::LinSpaced(8, 1., 8.);
    data.update_approximation(node_data::update_mode::eval_all, true);
    const matrix hessian = data.dense().lag_hess_[__x][__x].dense();
    matrix expected = matrix::Identity(8, 8);
    const scalar_t a = 3., b = 4., c = 5.;
    expected.block<3, 3>(2, 2) <<
        b * b + c * c, a * b, a * c,
        a * b, a * a + c * c, b * c,
        a * c, b * c, a * a + b * b;
    REQUIRE(hessian.isApprox(expected));
}

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

TEST_CASE("box inequalities preserve sliced and one-sided bounds") {
    auto [x, y] = sym::states("partial_box_x", 6);
    (void)y;
    const cs::SX selected = cs::SX::vertcat({x(1), x(4)});
    const vector lb = (vector(2) <<
        -std::numeric_limits<scalar_t>::infinity(), -2.).finished();
    const vector ub = (vector(2) <<
        1., std::numeric_limits<scalar_t>::infinity()).finished();
    auto c = ineq_constr::create("partial_box", var_inarg_list{},
                                 selected, lb, ub);
    const auto *box = dynamic_cast<const ineq_constr &>(*c).box_info();
    REQUIRE(box != nullptr);
    REQUIRE(box->base_dim == 2);
    REQUIRE(box->present_mask[box_side::lb].count() == 1);
    REQUIRE(box->present_mask[box_side::ub].count() == 1);
    REQUIRE(c->in_args().size() == 1);
    REQUIRE(c->in_args()[0].get() == x.get());

    auto prob = stage_ocp::create();
    prob->add(*c);
    prob->wait_until_ready();
    node_data data(prob);
    REQUIRE(data.dense().approx_[__ineq_x].jac_[__x].rows() == 2);
    REQUIRE(data.dense().approx_[__ineq_x].jac_[__x].cols() == 6);
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
    REQUIRE(hess.diag_panels_.size() == 1);
    REQUIRE(hess.diag_panels_[0].row_st_ == 0);
    REQUIRE(hess.diag_panels_[0].rows_ == 6);
}
} // namespace
