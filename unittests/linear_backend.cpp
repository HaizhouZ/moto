#include <catch2/catch_test_macros.hpp>

#include <moto/core/linear_backend.hpp>

#include <Eigen/Cholesky>
#include <moto/core/sparse_matrix.hpp>

#include <casadi/casadi.hpp>
#include <Eigen/LU>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>

namespace moto::linear_backend {

TEST_CASE("JIT panel products match dense algebra") {
  matrix dense = matrix::Zero(8, 7);
  matrix panel = matrix::Random(3, 2);
  vector diagonal = vector::Random(3);
  dense.block(0, 0, 3, 2) = panel;
  dense.block(3, 2, 3, 3).diagonal() = diagonal;
  dense.block(0, 5, 2, 2).diagonal().setOnes();
  matrix_layout layout{.rows = 8,
                       .cols = 7,
                       .panels = {{sparsity::dense, 0, 0, 3, 2},
                                  {sparsity::diag, 3, 2, 3, 3},
                                  {sparsity::eye, 0, 5, 2, 2}}};
  const auto check = [&](product_op op, const matrix &other,
                         const matrix &expected, scalar_t sign) {
    product_spec spec{.sparse = layout,
                      .op = op,
                      .other_rows = static_cast<size_t>(other.rows()),
                      .other_cols = static_cast<size_t>(other.cols()),
                      .out_rows = static_cast<size_t>(expected.rows()),
                      .out_cols = static_cast<size_t>(expected.cols()),
                      .sign = sign};
    auto kernel = compile_product(spec);
    matrix out = matrix::Zero(expected.rows(), expected.cols());
    std::vector<scalar_t *> pointers{panel.data(), diagonal.data(), nullptr,
                                     const_cast<scalar_t *>(other.data()),
                                     out.data()};
    kernel(pointers);
    REQUIRE(out.isApprox(sign * expected, 1e-12));
  };
  matrix right = matrix::Random(7, 5);
  check(product_op::times, right, dense * right, 1.);
  check(product_op::times, right, dense * right, -1.);
  matrix tright = matrix::Random(8, 5);
  check(product_op::transpose_times, tright, dense.transpose() * tright, 1.);
  matrix left = matrix::Random(4, 8);
  check(product_op::right_times, left, left * dense, 1.);
  matrix left_t = matrix::Random(8, 4);
  check(product_op::right_transpose_times, left_t, left_t.transpose() * dense,
        1.);
}

TEST_CASE("sparse_matrix dispatches dense operands through JIT") {
  sparse_matrix sparse;
  sparse.resize(6, 5);
  sparse.insert(0, 0, 2, 3, sparsity::dense).setRandom();
  sparse.insert(2, 2, 2, 2, sparsity::diag).setRandom();
  sparse.insert(4, 3, 2, 2, sparsity::eye);
  const matrix dense = sparse.dense();
  matrix rhs = matrix::Random(5, 4);
  matrix out = matrix::Zero(6, 4);
  linear_backend::multiply(sparse, rhs, out);
  REQUIRE(out.isApprox(dense * rhs, 1e-12));
  matrix trhs = matrix::Random(6, 3);
  matrix tout = matrix::Zero(5, 3);
  linear_backend::transpose_multiply(sparse, trhs, tout);
  REQUIRE(tout.isApprox(dense.transpose() * trhs, 1e-12));
  matrix lhs = matrix::Random(3, 6);
  matrix lout = matrix::Zero(3, 5);
  linear_backend::right_multiply(lhs, sparse, lout);
  REQUIRE(lout.isApprox(lhs * dense, 1e-12));
  matrix lhs_t = matrix::Random(6, 3);
  matrix ltout = matrix::Zero(3, 5);
  linear_backend::right_transpose_multiply(lhs_t, sparse, ltout);
  REQUIRE(ltout.isApprox(lhs_t.transpose() * dense, 1e-12));

  matrix sub = matrix::Random(6, 5), expected = sub - dense;
  write_dense(sparse, sub, {.alpha = -1.});
  REQUIRE(sub.isApprox(expected, 1e-12));
  sub.setRandom();
  expected = sub;
  for (Eigen::Index c = 0; c < dense.cols(); ++c)
    for (Eigen::Index r = 0; r < dense.rows(); ++r)
      if (dense(r, c) != 0.)
        expected(r, c) = -dense(r, c);
  write_dense(sparse, sub, {.alpha = -1., .overwrite = true});
  REQUIRE(sub.isApprox(expected, 1e-12));
}

TEST_CASE("precompiled direct-map sparse products match dense algebra") {
  const auto make_sparse = [](size_t rows, size_t cols) {
    sparse_matrix value;
    value.resize(rows, cols);
    const size_t diagonal = std::min(rows, cols);
    if (rows >= 3 && cols >= 3)
      value.insert(0, 0, 3, 3, sparsity::dense).setRandom();
    if (diagonal >= 6)
      value.insert(3, 2, 3, 3, sparsity::diag).setRandom();
    if (rows >= 2 && cols >= 2)
      value.insert(rows - 2, cols - 2, 2, 2, sparsity::eye);
    return value;
  };

  auto a = make_sparse(8, 7);
  auto rhs = make_sparse(7, 6);
  matrix out = matrix::Zero(8, 6);
  multiply(a, rhs, out);
  REQUIRE(out.isApprox(a.dense() * rhs.dense(), 1e-12));

  auto transpose_rhs = make_sparse(8, 6);
  matrix transpose_out = matrix::Zero(7, 6);
  transpose_multiply(a, transpose_rhs, transpose_out);
  REQUIRE(transpose_out.isApprox(a.dense().transpose() * transpose_rhs.dense(),
                                 1e-12));

  auto lhs = make_sparse(5, 8);
  matrix right_out = matrix::Zero(5, 7);
  right_multiply(lhs, a, right_out);
  REQUIRE(right_out.isApprox(lhs.dense() * a.dense(), 1e-12));

  auto transpose_lhs = make_sparse(8, 5);
  matrix right_transpose_out = matrix::Zero(5, 7);
  right_transpose_multiply(transpose_lhs, a, right_transpose_out);
  REQUIRE(right_transpose_out.isApprox(transpose_lhs.dense().transpose() *
                                           a.dense(),
                                       1e-12));
}

TEST_CASE("scaled eye panels use their dynamic diagonal values") {
  sparse_matrix sparse;
  sparse.resize(6, 6);
  auto eye = sparse.insert(0, 0, 6, 6, sparsity::eye);
  const vector scale = vector::LinSpaced(6, .25, 1.5);
  eye = scale;
  REQUIRE(sparse.set_dynamic_eye(true));
  REQUIRE(describe(sparse).panels.front().pattern == sparsity::diag);

  const matrix expected = scale.asDiagonal();
  matrix rhs = matrix::Random(6, 4), out = matrix::Zero(6, 4);
  multiply(sparse, rhs, out);
  REQUIRE(out.isApprox(expected * rhs, 1e-12));

  matrix trhs = matrix::Random(6, 3), transpose = matrix::Zero(6, 3);
  transpose_multiply(sparse, trhs, transpose);
  REQUIRE(transpose.isApprox(expected.transpose() * trhs, 1e-12));

  REQUIRE(sparse.dense().isApprox(expected, 1e-12));
}

TEST_CASE("OCP rowwise backend scales and reduces a sparse profile") {
  sparse_matrix sparse;
  sparse.resize(6, 8);
  auto dense = sparse.insert(0, 0, 3, 2, sparsity::dense);
  auto diagonal = sparse.insert(3, 2, 3, 3, sparsity::diag);
  auto eye = sparse.insert(0, 5, 3, 3, sparsity::eye);
  dense.setRandom();
  diagonal.setRandom();
  eye.setOnes();
  sparse.set_dynamic_eye(true);
  const auto layout = describe(sparse);
  const auto pointers = panel_pointers(sparse);

  vector norms = vector::Random(6).cwiseAbs();
  vector expected = norms;
  const matrix before = sparse.dense();
  expected = expected.cwiseMax(before.cwiseAbs().rowwise().maxCoeff());
  const auto kernels = compile_rowwise(layout);
  kernels.inf_norm(pointers, nullptr, norms.data());
  REQUIRE(norms.isApprox(expected, 1e-12));

  const vector scale = vector::LinSpaced(6, .5, 1.5);
  vector scaled_norms = vector::Zero(6);
  const vector expected_scaled =
      scale.cwiseAbs().cwiseProduct(before.cwiseAbs().rowwise().maxCoeff());
  kernels.scaled_inf_norm(pointers, scale.data(), scaled_norms.data());
  REQUIRE(scaled_norms.isApprox(expected_scaled, 1e-12));

  kernels.scale(pointers, scale.data(), nullptr);
  REQUIRE(sparse.dense().isApprox(scale.asDiagonal() * before, 1e-12));
}

TEST_CASE("backend preserves aligned panel storage") {
  sparse_matrix sparse;
  sparse.resize(17, 13);
  auto dense = sparse.insert(0, 0, 7, 5, sparsity::dense);
  auto diagonal = sparse.insert(7, 5, 6, 6, sparsity::diag);
  const auto aligned = [](const scalar_t *pointer) {
    return reinterpret_cast<std::uintptr_t>(pointer) % EIGEN_MAX_ALIGN_BYTES ==
           0;
  };
  REQUIRE(aligned(dense.data()));
  REQUIRE(aligned(diagonal.data()));

  // Sliced structured panels are intentionally unaligned; the backend must
  // keep their base allocation aligned and use unaligned Maps for offsets.
  REQUIRE_NOTHROW(Eigen::Map<const vector>(diagonal.data() + 1, 5).sum());
}

TEST_CASE("JIT fuses overlapping diagonal contributions") {
  sparse_matrix sparse;
  sparse.resize(18, 18);
  auto d0 = sparse.insert(0, 0, 18, 18, sparsity::diag);
  auto d1 = sparse.insert(0, 0, 18, 18, sparsity::diag);
  d0.setRandom();
  d1.setRandom();
  vector rhs = vector::Random(18), out = vector::Zero(18);
  multiply(sparse, rhs, out);
  REQUIRE(out.isApprox((d0 + d1).asDiagonal() * rhs, 1e-12));
  matrix rhs_m = matrix::Random(18, 7), out_m = matrix::Zero(18, 7);
  transpose_multiply(sparse, rhs_m, out_m);
  REQUIRE(out_m.isApprox((d0 + d1).asDiagonal() * rhs_m, 1e-12));
  matrix lhs = matrix::Random(5, 18), right = matrix::Zero(5, 18);
  right_multiply(lhs, sparse, right);
  REQUIRE(right.isApprox(lhs * (d0 + d1).asDiagonal(), 1e-12));
  matrix lhs_t = matrix::Random(18, 5), right_t = matrix::Zero(5, 18);
  right_transpose_multiply(lhs_t, sparse, right_t);
  REQUIRE(right_t.isApprox(lhs_t.transpose() * (d0 + d1).asDiagonal(), 1e-12));
  REQUIRE(sparse.dense().diagonal().isApprox(d0 + d1, 1e-12));
}

TEST_CASE("static profile physically fuses adjacent diagonal bindings") {
  sparse_matrix sparse;
  sparse.resize(7, 7);
  const std::array blocks{
      sparse_block_spec{0, 0, 3, 3, sparsity::diag},
      sparse_block_spec{3, 3, 4, 4, sparsity::diag}};
  sparse.plan(blocks);
  auto d0 = sparse.insert(0, 0, 3, 3, sparsity::diag);
  auto d1 = sparse.insert(3, 3, 4, 4, sparsity::diag);
  d0.setRandom();
  d1.setRandom();
  REQUIRE(sparse.diag_panels_.size() == 1);
  REQUIRE(d1.data() == d0.data() + d0.size());
  vector expected(7);
  expected << d0, d1;
  REQUIRE(sparse.dense().diagonal().isApprox(expected, 1e-12));
}

TEST_CASE("static profile packs disjoint diagonal panels into one owner") {
  sparse_matrix sparse;
  sparse.resize(12, 9);
  const std::array blocks{
      sparse_block_spec{0, 0, 3, 3, sparsity::diag},
      sparse_block_spec{5, 5, 4, 4, sparsity::diag},
      sparse_block_spec{9, 0, 3, 3, sparsity::diag}};
  auto layout = make_sparse_layout_plan(blocks);
  layout.pack_diagonal_storage = true;
  sparse.plan(layout);
  auto d0 = sparse.bind(0, 0, 3, 3, sparsity::diag);
  auto d1 = sparse.bind(5, 5, 4, 4, sparsity::diag);
  auto d2 = sparse.bind(9, 0, 3, 3, sparsity::diag);
  REQUIRE(sparse.diag_panels_.size() == 1);
  REQUIRE(sparse.diagonal_segments_.size() == 3);
  const auto aligned = [](const scalar_t *pointer) {
    return reinterpret_cast<std::uintptr_t>(pointer) % EIGEN_MAX_ALIGN_BYTES ==
           0;
  };
  REQUIRE(aligned(d0.data()));
  REQUIRE(aligned(d1.data()));
  REQUIRE(aligned(d2.data()));
  REQUIRE(d1.data() >= d0.data() + d0.size());
  REQUIRE(d2.data() >= d1.data() + d1.size());

  d0.setRandom();
  d1.setRandom();
  d2.setRandom();
  matrix expected = matrix::Zero(12, 9);
  expected.block(0, 0, 3, 3).diagonal() = d0;
  expected.block(5, 5, 4, 4).diagonal() = d1;
  expected.block(9, 0, 3, 3).diagonal() = d2;
  REQUIRE(sparse.dense().isApprox(expected, 1e-12));
  matrix rhs = matrix::Random(9, 4), out = matrix::Zero(12, 4);
  multiply(sparse, rhs, out);
  REQUIRE(out.isApprox(expected * rhs, 1e-12));
}

TEST_CASE("sparse matrix resize accepts panels ending at the boundary") {
  sparse_matrix dense;
  dense.resize(4, 5);
  dense.insert(0, 0, 4, 5, sparsity::dense);
  dense.resize(4, 5);

  sparse_matrix diagonal;
  diagonal.resize(4, 4);
  diagonal.insert(0, 0, 4, 4, sparsity::diag);
  diagonal.resize(4, 4);

  sparse_matrix eye;
  eye.resize(4, 4);
  eye.insert(0, 0, 4, 4, sparsity::eye);
  eye.resize(4, 4);
}

TEST_CASE("sparse matrix copies own independent JIT bindings") {
  sparse_matrix source;
  source.resize(3, 3);
  source.insert(0, 0, 3, 3, sparsity::dense).setOnes();
  const matrix rhs = matrix::Identity(3, 3);
  matrix output = matrix::Zero(3, 3);
  multiply(source, rhs, output);

  sparse_matrix copy(source);
  copy.dense_panels_.front().data_.setConstant(2.);
  output.setZero();
  multiply(copy, rhs, output);
  REQUIRE(output.isConstant(2.));
  REQUIRE(source.dense().isConstant(1.));
}

TEST_CASE("sparse matrix assignment preserves an unused static binding plan") {
  sparse_matrix source;
  source.resize(4, 4);
  const std::array blocks{
      sparse_block_spec{0, 0, 4, 4, sparsity::diag}};
  source.plan(blocks);

  sparse_matrix copy;
  copy = source;
  copy.bind(0, 0, 4, 4, sparsity::diag).setOnes();
  REQUIRE(copy.dense().diagonal().isOnes());
}

TEST_CASE("additive layout aliases exactly overlapping diagonal bindings") {
  sparse_matrix sparse;
  sparse.resize(6, 6);
  const std::array blocks{
      sparse_block_spec{0, 0, 6, 6, sparsity::diag},
      sparse_block_spec{0, 0, 6, 6, sparsity::diag}};
  const auto layout =
      make_sparse_layout_plan(blocks, sparse_plan_mode::additive);
  sparse.plan(layout);
  auto d0 = sparse.bind(0, 0, 6, 6, sparsity::diag);
  auto d1 = sparse.bind(0, 0, 6, 6, sparsity::diag);
  REQUIRE(sparse.diag_panels_.size() == 1);
  REQUIRE(d0.data() == d1.data());
  sparse.setZero();
  d0.array() += 2.;
  d1.array() += 3.;
  REQUIRE(sparse.dense().diagonal().isConstant(5.));
}

TEST_CASE("additive layout aliases exactly overlapping dense bindings") {
  sparse_matrix sparse;
  sparse.resize(5, 4);
  const std::array blocks{
      sparse_block_spec{0, 0, 5, 4, sparsity::dense},
      sparse_block_spec{0, 0, 5, 4, sparsity::dense}};
  sparse.plan(make_sparse_layout_plan(blocks, sparse_plan_mode::additive));
  auto a = sparse.bind(0, 0, 5, 4, sparsity::dense);
  auto b = sparse.bind(0, 0, 5, 4, sparsity::dense);
  REQUIRE(sparse.dense_panels_.size() == 1);
  REQUIRE(a.data() == b.data());
  sparse.setZero();
  a.array() += 2.;
  b.array() += 3.;
  REQUIRE(sparse.dense().isConstant(5.));
}

TEST_CASE("identity Hessian shares the additive base diagonal") {
  sparse_matrix sparse;
  sparse.resize(5, 5);
  const std::array blocks{
      sparse_block_spec{0, 0, 5, 5, sparsity::diag},
      sparse_block_spec{0, 0, 5, 5, sparsity::eye}};
  sparse.plan(make_sparse_layout_plan(blocks, sparse_plan_mode::additive));
  auto e0 = sparse.bind(0, 0, 5, 5, sparsity::diag);
  auto e1 = sparse.bind(0, 0, 5, 5, sparsity::eye);
  REQUIRE(sparse.eye_panels_.empty());
  REQUIRE(sparse.diag_panels_.size() == 1);
  REQUIRE(e0.data() == e1.data());
  sparse.setZero();
  e0.array() += 1.;
  e1.array() += 1.;
  REQUIRE(sparse.dense().diagonal().isConstant(2.));
}

TEST_CASE("distinct layout preserves exact overlaps and bind is strict") {
  sparse_matrix sparse;
  sparse.resize(4, 4);
  const std::array blocks{
      sparse_block_spec{0, 0, 4, 4, sparsity::diag},
      sparse_block_spec{0, 0, 4, 4, sparsity::diag}};
  sparse.plan(blocks);
  auto d0 = sparse.bind(0, 0, 4, 4, sparsity::diag);
  auto d1 = sparse.bind(0, 0, 4, 4, sparsity::diag);
  REQUIRE(sparse.diag_panels_.size() == 2);
  REQUIRE(d0.data() != d1.data());
  REQUIRE_THROWS_AS(sparse.bind(0, 0, 4, 4, sparsity::diag),
                    std::logic_error);
}

TEST_CASE("planned references survive fallback panel growth") {
  sparse_matrix sparse;
  sparse.resize(4, 4);
  const std::array blocks{
      sparse_block_spec{0, 0, 4, 4, sparsity::diag}};
  sparse.plan(blocks);
  auto planned = sparse.insert(0, 0, 4, 4, sparsity::diag);
  for (size_t i = 0; i < 16; ++i)
    sparse.insert(0, 0, 4, 4, sparsity::diag).setZero();
  planned.setOnes();
  REQUIRE(sparse.dense().diagonal().isOnes());
}

TEST_CASE("JIT fuses overlapping dense contributions") {
  sparse_matrix sparse;
  sparse.resize(8, 6);
  auto a = sparse.insert(0, 0, 8, 6, sparsity::dense);
  auto b = sparse.insert(0, 0, 8, 6, sparsity::dense);
  a.setRandom();
  b.setRandom();
  matrix rhs = matrix::Random(6, 5), out = matrix::Zero(8, 5);
  multiply(sparse, rhs, out);
  REQUIRE(out.isApprox((a + b) * rhs, 1e-12));
}

TEST_CASE("OCP batch fuses quadruped limits and friction") {
  const panel_layout q{.pattern = sparsity::eye,
                       .row_offset = 0,
                       .col_offset = 7,
                       .rows = 12,
                       .cols = 12};
  const panel_layout v{.pattern = sparsity::eye,
                       .row_offset = 12,
                       .col_offset = 6,
                       .rows = 12,
                       .cols = 12};
  const panel_layout tq{.pattern = sparsity::eye,
                        .row_offset = 0,
                        .col_offset = 0,
                        .rows = 12,
                        .cols = 12};
  const panel_layout fc{.pattern = sparsity::dense,
                        .row_offset = 0,
                        .col_offset = 0,
                        .rows = 4,
                        .cols = 3};
  batch_condensation_spec batch{{
      {.rows = 24, .jacobians = {q, v}, .residual_signs = {1., -1.}},
      {.rows = 12, .jacobians = {tq}, .residual_signs = {1., -1.}},
      {.rows = 4, .jacobians = {fc}, .residual_signs = {1.}},
  }};
  auto kernel = compile_batch_condensation(batch);

  vector qr0 = vector::Random(24), qr1 = vector::Random(24);
  vector qw0 = vector::Random(24).cwiseAbs();
  vector qw1 = vector::Random(24).cwiseAbs();
  row_vector gq = row_vector::Random(19), gv = row_vector::Random(18);
  vector hqq = vector::Random(12), hqv = vector::Random(12);
  vector hvv = vector::Random(12);
  row_vector expected_gq = gq, expected_gv = gv;
  expected_gq.segment(7, 12) += (qr0 - qr1).head(12).transpose();
  expected_gv.segment(6, 12) += (qr0 - qr1).tail(12).transpose();
  vector expected_hqq = hqq + (qw0 + qw1).head(12);
  vector expected_hqv = hqv;
  vector expected_hvv = hvv + (qw0 + qw1).tail(12);

  vector tr0 = vector::Random(12), tr1 = vector::Random(12);
  vector tw0 = vector::Random(12).cwiseAbs();
  vector tw1 = vector::Random(12).cwiseAbs();
  row_vector gt = row_vector::Random(12);
  vector ht = vector::Random(12);
  row_vector expected_gt = gt + (tr0 - tr1).transpose();
  vector expected_ht = ht + tw0 + tw1;

  matrix jf = matrix::Random(4, 3);
  vector fr = vector::Random(4), fw = vector::Random(4).cwiseAbs();
  row_vector gf = row_vector::Random(3);
  matrix hf = matrix::Random(3, 3);
  row_vector expected_gf = gf + fr.transpose() * jf;
  matrix expected_hf = hf + jf.transpose() * fw.asDiagonal() * jf;

  std::vector<scalar_t *> p{
      nullptr,    nullptr,    qr0.data(), qr1.data(), qw0.data(), qw1.data(),
      gq.data(),  gv.data(),  hqq.data(), hqv.data(), hvv.data(), nullptr,
      tr0.data(), tr1.data(), tw0.data(), tw1.data(), gt.data(),  ht.data(),
      jf.data(),  fr.data(),  fw.data(),  gf.data(),  hf.data()};
  kernel(p);
  REQUIRE(gq.isApprox(expected_gq, 1e-12));
  REQUIRE(gv.isApprox(expected_gv, 1e-12));
  REQUIRE(hqq.isApprox(expected_hqq, 1e-12));
  REQUIRE(hqv.isApprox(expected_hqv, 1e-12));
  REQUIRE(hvv.isApprox(expected_hvv, 1e-12));
  REQUIRE(gt.isApprox(expected_gt, 1e-12));
  REQUIRE(ht.isApprox(expected_ht, 1e-12));
  REQUIRE(gf.isApprox(expected_gf, 1e-12));
  REQUIRE(hf.isApprox(expected_hf, 1e-12));

  vector dxq = vector::Random(19), dxv = vector::Random(18);
  vector dxt = vector::Random(12), dxf = vector::Random(3);
  vector jqdx(24), jtdx(12), jfdx(4);
  auto jv_kernel =
      compile_batch_jacobian_product({{24, {q, v}}, {12, {tq}}, {4, {fc}}});
  std::vector<scalar_t *> jp{nullptr,     nullptr,    dxq.data(), dxv.data(),
                             jqdx.data(), nullptr,    dxt.data(), jtdx.data(),
                             jf.data(),   dxf.data(), jfdx.data()};
  jv_kernel(jp);
  vector expected_jqdx(24);
  expected_jqdx << dxq.segment(7, 12), dxv.segment(6, 12);
  REQUIRE(jqdx.isApprox(expected_jqdx, 1e-12));
  REQUIRE(jtdx.isApprox(dxt, 1e-12));
  REQUIRE(jfdx.isApprox(jf * dxf, 1e-12));
}

TEST_CASE("MX graph lowers matrix products to the linear backend") {
  constexpr casadi_int n = 4, middle = 3, cols = 2;
  const casadi::MX a = casadi::MX::sym("a", casadi::Sparsity::diag(n));
  const casadi::MX b = casadi::MX::sym("b", n, middle);
  const casadi::MX c = casadi::MX::sym("c", n, middle);
  const casadi::MX d = casadi::MX::sym("d", middle, cols);
  const casadi::MX first = casadi::MX::mtimes(a, b) + c;
  const casadi::MX second = casadi::MX::mtimes(first, d);
  const auto kernel = compile_graph({a, b, c, d}, {first, second});

  vector av = vector::Random(n);
  matrix bv = matrix::Random(n, middle), cv = matrix::Random(n, middle);
  matrix dv = matrix::Random(middle, cols);
  matrix first_value(n, middle), second_value(n, cols);
  std::vector<scalar_t *> pointers{av.data(), bv.data(), cv.data(), dv.data(),
                                   first_value.data(), second_value.data()};
  kernel(pointers);
  const matrix expected_first = av.asDiagonal() * bv + cv;
  REQUIRE(first_value.isApprox(expected_first, 1e-12));
  REQUIRE(second_value.isApprox(expected_first * dv, 1e-12));
}

TEST_CASE("MX graph lowers a dense matrix chain") {
  constexpr casadi_int outer = 20, thin = 2;
  const casadi::MX a = casadi::MX::sym("chain_a", outer, thin);
  const casadi::MX b = casadi::MX::sym("chain_b", thin, outer);
  const casadi::MX c = casadi::MX::sym("chain_c", outer, thin);
  const casadi::MX original =
      casadi::MX::mtimes(casadi::MX::mtimes(a, b), c);
  const auto kernel = compile_graph({a, b, c}, {original});

  const matrix av = matrix::Random(outer, thin);
  const matrix bv = matrix::Random(thin, outer);
  const matrix cv = matrix::Random(outer, thin);
  matrix output(outer, thin);
  std::vector<scalar_t *> pointers{
      const_cast<scalar_t *>(av.data()), const_cast<scalar_t *>(bv.data()),
      const_cast<scalar_t *>(cv.data()), output.data()};
  kernel(pointers);
  REQUIRE(output.isApprox(av * (bv * cv), 1e-12));
}

TEST_CASE("MX graph lowers submatrix products to backend panel views") {
  const casadi::MX storage = casadi::MX::sym("view_storage", 6, 5);
  const casadi::MX rhs = casadi::MX::sym("view_rhs", 4, 2);
  const casadi::MX transpose_rhs =
      casadi::MX::sym("view_transpose_rhs", 3, 2);
  const casadi::MX block =
      storage(casadi::Slice(1, 4), casadi::Slice(1, 5));
  const auto kernel = compile_graph(
      {storage, rhs, transpose_rhs},
      {casadi::MX::mtimes(block, rhs),
       casadi::MX::mtimes(block.T(), transpose_rhs)});

  matrix storage_value = matrix::Random(6, 5);
  matrix rhs_value = matrix::Random(4, 2);
  matrix transpose_rhs_value = matrix::Random(3, 2);
  matrix output(3, 2), transpose_output(4, 2);
  std::vector<scalar_t *> pointers{
      storage_value.data(), rhs_value.data(), transpose_rhs_value.data(),
      output.data(), transpose_output.data()};
  kernel(pointers);
  REQUIRE(output.isApprox(
      storage_value.block(1, 1, 3, 4) * rhs_value, 1e-12));
  REQUIRE(transpose_output.isApprox(
      storage_value.block(1, 1, 3, 4).transpose() * transpose_rhs_value,
      1e-12));
}

TEST_CASE("MX lazy product fusion preserves intermediate addend order") {
  constexpr casadi_int n = 5;
  const casadi::MX a = casadi::MX::sym("ordered_a", n, n);
  const casadi::MX b = casadi::MX::sym("ordered_b", n, n);
  const casadi::MX c = casadi::MX::sym("ordered_c", n, n);
  const casadi::MX d = casadi::MX::sym("ordered_d", n, n);
  const casadi::MX lhs = casadi::MX::mtimes(a, b);
  const casadi::MX addend = casadi::MX::mtimes(c, d);
  const auto kernel = compile_graph(
      {a, b, c, d}, {casadi::MX::densify(lhs + addend)});

  matrix av = matrix::Random(n, n), bv = matrix::Random(n, n);
  matrix cv = matrix::Random(n, n), dv = matrix::Random(n, n);
  matrix output(n, n);
  std::vector<scalar_t *> pointers{av.data(), bv.data(), cv.data(),
                                   dv.data(), output.data()};
  kernel(pointers);
  REQUIRE(output.isApprox(av * bv + cv * dv, 1e-12));
}

TEST_CASE("MX product accumulation chains share one result") {
  constexpr casadi_int n = 5;
  const casadi::MX a = casadi::MX::sym("accum_a", n, n);
  const casadi::MX b = casadi::MX::sym("accum_b", n, n);
  const casadi::MX c = casadi::MX::sym("accum_c", n, n);
  const casadi::MX d = casadi::MX::sym("accum_d", n, n);
  const casadi::MX e = casadi::MX::sym("accum_e", n, n);
  const casadi::MX f = casadi::MX::sym("accum_f", n, n);
  const casadi::MX diagonal =
      casadi::MX::sym("accum_diagonal", casadi::Sparsity::diag(n));
  const casadi::MX result =
      casadi::MX::densify(diagonal) + casadi::MX::mtimes(a, b) -
      casadi::MX::mtimes(c, d) + casadi::MX::mtimes(e, f);
  const auto kernel = compile_graph({a, b, c, d, e, f, diagonal}, {result});

  matrix av = matrix::Random(n, n), bv = matrix::Random(n, n);
  matrix cv = matrix::Random(n, n), dv = matrix::Random(n, n);
  matrix ev = matrix::Random(n, n), fv = matrix::Random(n, n);
  vector diagonal_value = vector::Random(n);
  matrix output(n, n);
  std::vector<scalar_t *> pointers{
      av.data(), bv.data(), cv.data(), dv.data(), ev.data(), fv.data(),
      diagonal_value.data(), output.data()};
  kernel(pointers);
  matrix expected = diagonal_value.asDiagonal();
  expected.noalias() += av * bv;
  expected.noalias() -= cv * dv;
  expected.noalias() += ev * fv;
  REQUIRE(output.isApprox(expected, 1e-12));
}

TEST_CASE("MX graph lowers sparse views without a CasADi runtime") {
  const casadi::Sparsity pattern = casadi::Sparsity::triplet(
      4, 3, std::vector<casadi_int>{0, 2, 3},
      std::vector<casadi_int>{0, 1, 2});
  const casadi::MX sparse = casadi::MX::sym("sparse", pattern);
  const casadi::MX repeated = casadi::MX::repmat(sparse, 1, 2);
  const auto kernel = compile_graph(
      {sparse}, {casadi::MX::densify(sparse),
                 casadi::MX::densify(sparse.T()),
                 casadi::MX::densify(repeated)});

  vector values = vector::Random(pattern.nnz());
  matrix dense = matrix::Zero(4, 3), dense_output(4, 3), transpose_output(3, 4),
         repeat_output(4, 6);
  for (casadi_int col = 0; col < pattern.size2(); ++col)
    for (casadi_int nz = pattern.colind(col); nz < pattern.colind(col + 1);
         ++nz)
      dense(pattern.row(nz), col) = values[nz];
  std::vector<scalar_t *> pointers{values.data(), dense_output.data(),
                                   transpose_output.data(),
                                   repeat_output.data()};
  kernel(pointers);
  matrix expected_repeat(4, 6);
  expected_repeat << dense, dense;
  REQUIRE(dense_output.isApprox(dense, 1e-12));
  REQUIRE(transpose_output.isApprox(dense.transpose(), 1e-12));
  REQUIRE(repeat_output.isApprox(expected_repeat, 1e-12));
}

TEST_CASE("MX graph binds one logical input to existing sparse panels") {
  const casadi::Sparsity pattern = casadi::Sparsity::triplet(
      4, 4, std::vector<casadi_int>{0, 1, 0, 1, 2, 3},
      std::vector<casadi_int>{0, 0, 1, 1, 2, 3});
  const casadi::MX sparse = casadi::MX::sym("panel_sparse", pattern);
  const casadi::MX rhs = casadi::MX::sym("panel_rhs", 4, 2);
  matrix_layout layout{
      4, 4,
      {{sparsity::dense, 0, 0, 2, 2},
       {sparsity::diag, 2, 2, 2, 2}}};
  const std::array<matrix_layout, 2> input_layouts{layout, matrix_layout{}};
  const auto kernel = compile_graph(
      {sparse, rhs},
      std::vector<std::vector<casadi::MX>>{
          {casadi::MX::densify(casadi::MX::mtimes(sparse, rhs))}},
      input_layouts, nullptr);

  matrix dense_panel = matrix::Random(2, 2);
  vector diagonal = vector::Random(2);
  matrix rhs_value = matrix::Random(4, 2), output(4, 2);
  std::vector<scalar_t *> pointers{dense_panel.data(), diagonal.data(),
                                   rhs_value.data(), output.data()};
  kernel(pointers);

  matrix expected_matrix = matrix::Zero(4, 4);
  expected_matrix.topLeftCorner(2, 2) = dense_panel;
  expected_matrix.bottomRightCorner(2, 2) = diagonal.asDiagonal();
  REQUIRE(kernel.input_count() == 3);
  REQUIRE(output.isApprox(expected_matrix * rhs_value, 1e-12));
}

TEST_CASE("MX graph lowers a multi-RHS solve as one scheduled operation") {
  constexpr casadi_int n = 5, rhs_cols = 2;
  const casadi::MX a = casadi::MX::sym("a", n, n);
  const casadi::MX b = casadi::MX::sym("b", n, rhs_cols);
  const auto kernel = compile_graph({a, b}, {casadi::MX::solve(a, b)});
  matrix av = matrix::Random(n, n);
  av.diagonal().array() += 4.;
  matrix bv = matrix::Random(n, rhs_cols), output(n, rhs_cols);
  std::vector<scalar_t *> pointers{av.data(), bv.data(), output.data()};
  kernel(pointers);
  REQUIRE(output.isApprox(av.partialPivLu().solve(bv), 1e-12));
}

TEST_CASE("MX graph lazily applies one shared factor to products and actions") {
  constexpr casadi_int n = 5, cols = 3;
  const casadi::MX a = casadi::MX::sym("inverse_a", n, n);
  const casadi::MX b = casadi::MX::sym("inverse_b", n, cols);
  const casadi::MX c = casadi::MX::sym("inverse_c", n, cols);
  const casadi::MX inverse =
      casadi::MX::solve(a, casadi::MX::eye(n));
  const auto kernel = compile_graph(
      {a, b, c},
      std::vector<std::vector<casadi::MX>>{
          {casadi::MX::mtimes(inverse, b)},
          {casadi::MX::mtimes(inverse.T(), c)}},
      nullptr);

  matrix av = matrix::Random(n, n);
  av.diagonal().array() += 5.;
  matrix bv = matrix::Random(n, cols), cv = matrix::Random(n, cols);
  const matrix expected_inverse = av.inverse();
  matrix first(n, cols), second(n, cols);
  std::vector<scalar_t *> pointers{av.data(), bv.data(), cv.data(),
                                   first.data(), second.data()};

  kernel(0, pointers);
  REQUIRE(first.isApprox(expected_inverse * bv, 1e-12));
  kernel(1, pointers);
  REQUIRE(second.isApprox(expected_inverse.transpose() * cv, 1e-12));
}

TEST_CASE("MX graph preserves an SPD factor declaration across graph CSE") {
  constexpr casadi_int n = 5, cols = 3;
  const casadi::MX a = casadi::MX::sym("spd_a", n, n);
  const casadi::MX b = casadi::MX::sym("spd_b", n, cols);
  const casadi::MX inverse = casadi::MX::inv(a);
  const std::array<casadi::MX, 1> spd{inverse};
  const auto kernel = compile_graph(
      {}, {a, b},
      std::vector<std::vector<casadi::MX>>{
          {casadi::MX::mtimes(inverse, b)}},
      {}, nullptr, "gen/linear_backend", spd);

  const matrix seed = matrix::Random(n, n);
  const matrix av = seed.transpose() * seed + matrix::Identity(n, n);
  const matrix bv = matrix::Random(n, cols);
  matrix output(n, cols);
  std::vector<scalar_t *> pointers{
      const_cast<scalar_t *>(av.data()), const_cast<scalar_t *>(bv.data()),
      output.data()};
  kernel(pointers);
  REQUIRE(output.isApprox(av.llt().solve(bv), 1e-12));
}

TEST_CASE("MX graph directly solves fixed matrices up to three by three") {
  constexpr casadi_int n = 3, cols = 2;
  const casadi::MX a = casadi::MX::sym("small_a", n, n);
  const casadi::MX b = casadi::MX::sym("small_b", n, cols);
  const auto kernel = compile_graph({a, b}, {casadi::MX::solve(a, b)});

  matrix av = matrix::Random(n, n);
  av.diagonal().array() += 4.;
  const matrix bv = matrix::Random(n, cols);
  matrix output(n, cols);
  std::vector<scalar_t *> pointers{
      av.data(), const_cast<scalar_t *>(bv.data()), output.data()};
  kernel(pointers);
  REQUIRE(output.isApprox(av.partialPivLu().solve(bv), 1e-12));
}

TEST_CASE("Go2-sized elimination graph matches one fused Eigen routine") {
  constexpr casadi_int nq = 18, ny = 36, nl = 30, nf = 12;
  constexpr casadi_int nx_active = 30, nu = 18, nr = 1;
  constexpr casadi_int rhs_cols = nx_active + nu + nr;

  std::vector<casadi_int> rows, cols;
  const auto add_diag = [&](casadi_int begin, casadi_int count) {
    for (casadi_int i = 0; i < count; ++i) {
      rows.push_back(begin + i);
      cols.push_back(begin + i);
    }
  };
  add_diag(0, 3);
  for (casadi_int col = 3; col < 6; ++col)
    for (casadi_int row = 3; row < 6; ++row) {
      rows.push_back(row);
      cols.push_back(col);
    }
  add_diag(6, 12);
  for (casadi_int col = nq; col < ny; ++col)
    for (casadi_int row = 0; row < nq; ++row) {
      rows.push_back(row);
      cols.push_back(col);
    }
  add_diag(nq, nq);
  const casadi::Sparsity euler_pattern =
      casadi::Sparsity::triplet(ny, ny, rows, cols);

  const casadi::MX euler = casadi::MX::sym("bench_euler", euler_pattern);
  const casadi::MX dyn_l = casadi::MX::sym("bench_dyn_l", ny, nl);
  const casadi::MX lift_y = casadi::MX::sym("bench_lift_y", nl, ny);
  const casadi::MX lift_l = casadi::MX::sym("bench_lift_l", nl, nl);
  const casadi::MX h_x = casadi::MX::sym("bench_h_x", ny + nl, nx_active);
  const casadi::MX h_u = casadi::MX::sym("bench_h_u", ny + nl, nu);
  const casadi::MX h_r = casadi::MX::sym("bench_h_r", ny + nl, nr);

  const auto euler_solve = [&](const casadi::MX &rhs) {
    const casadi::MX q_rhs =
        rhs(casadi::Slice(0, nq), casadi::Slice()) -
        casadi::MX::mtimes(
            euler(casadi::Slice(0, nq), casadi::Slice(nq, ny)),
            rhs(casadi::Slice(nq, ny), casadi::Slice()));
    const casadi::MX diagonal = casadi::MX::diag(euler);
    const auto diagonal_solve = [&](casadi_int begin, casadi_int end) {
      return q_rhs(casadi::Slice(begin, end), casadi::Slice()) /
             casadi::MX::repmat(
                 diagonal(casadi::Slice(begin, end), casadi::Slice()), 1,
                 rhs.size2());
    };
    return casadi::MX::vertcat(
        {diagonal_solve(0, 3),
         casadi::MX::solve(
             euler(casadi::Slice(3, 6), casadi::Slice(3, 6)),
             q_rhs(casadi::Slice(3, 6), casadi::Slice())),
         diagonal_solve(6, nq),
         rhs(casadi::Slice(nq, ny), casadi::Slice())});
  };

  const casadi::MX euler_l = euler_solve(dyn_l);
  const casadi::MX reduced_l =
      lift_l - casadi::MX::mtimes(lift_y, euler_l);
  const casadi::MX gaa =
      reduced_l(casadi::Slice(0, nq), casadi::Slice(0, nq));
  const casadi::MX gaf =
      reduced_l(casadi::Slice(0, nq), casadi::Slice(nq, nl));
  const casadi::MX gca =
      reduced_l(casadi::Slice(nq, nl), casadi::Slice(0, nq));
  const casadi::MX gcf =
      reduced_l(casadi::Slice(nq, nl), casadi::Slice(nq, nl)) +
      1e-3 * casadi::MX::eye(nf);
  const casadi::MX rnea_force = casadi::MX::solve(gaa, gaf);
  const casadi::MX contact_schur =
      gcf - casadi::MX::mtimes(gca, rnea_force);
  const casadi::MX rhs = casadi::MX::horzcat({h_x, h_u, h_r});
  const casadi::MX y_base = euler_solve(
      rhs(casadi::Slice(0, ny), casadi::Slice()));
  const casadi::MX reduced =
      rhs(casadi::Slice(ny, ny + nl), casadi::Slice()) -
      casadi::MX::mtimes(lift_y, y_base);
  const casadi::MX a_base = casadi::MX::solve(
      gaa, reduced(casadi::Slice(0, nq), casadi::Slice()));
  const casadi::MX force = casadi::MX::solve(
      contact_schur,
      reduced(casadi::Slice(nq, nl), casadi::Slice()) -
          casadi::MX::mtimes(gca, a_base));
  const casadi::MX lifted = casadi::MX::vertcat(
      {a_base - casadi::MX::mtimes(rnea_force, force), force});
  const casadi::MX result = casadi::MX::vertcat(
      {y_base - casadi::MX::mtimes(euler_l, lifted), lifted});

  const matrix_layout euler_layout{
      ny,
      ny,
      {{sparsity::diag, 0, 0, 3, 3},
       {sparsity::dense, 3, 3, 3, 3},
       {sparsity::diag, 6, 6, 12, 12},
       {sparsity::dense, 0, nq, nq, nq},
       {sparsity::eye, nq, nq, nq, nq}}};
  const std::array<matrix_layout, 7> input_layouts{
      euler_layout, matrix_layout{}, matrix_layout{}, matrix_layout{},
      matrix_layout{}, matrix_layout{}, matrix_layout{}};
  auto graph = compile_graph(
      {euler, dyn_l, lift_y, lift_l, h_x, h_u, h_r},
      std::vector<std::vector<casadi::MX>>{{result}}, input_layouts, nullptr);

  vector q_diag_head = vector::Random(3).cwiseAbs().array() + 1.;
  matrix orientation = matrix::Random(3, 3);
  orientation.diagonal().array() += 4.;
  vector q_diag_tail = vector::Random(12).cwiseAbs().array() + 1.;
  matrix coupling = matrix::Random(nq, nq) * .05;
  matrix dyn_l_value = matrix::Random(ny, nl) * .05;
  matrix lift_y_value = matrix::Random(nl, ny) * .05;
  matrix lift_l_value = matrix::Random(nl, nl) * .05;
  lift_l_value.topLeftCorner(nq, nq).diagonal().array() += 6.;
  lift_l_value.bottomRightCorner(nf, nf).diagonal().array() += 5.;
  matrix hx_value = matrix::Random(ny + nl, nx_active) * .05;
  matrix hu_value = matrix::Random(ny + nl, nu) * .05;
  matrix hr_value = matrix::Random(ny + nl, nr) * .05;
  matrix graph_output(ny + nl, rhs_cols);
  scalar_t unused_eye_storage = 0.;
  std::vector<scalar_t *> pointers{
      q_diag_head.data(), orientation.data(), q_diag_tail.data(),
      coupling.data(), &unused_eye_storage, dyn_l_value.data(),
      lift_y_value.data(), lift_l_value.data(), hx_value.data(),
      hu_value.data(), hr_value.data(), graph_output.data()};
  // The exact input layout contributes five physical pointers in place of
  // the first logical MX input.
  REQUIRE(pointers.size() == graph.pointer_count());

  struct fused_workspace {
    fused_workspace(casadi_int nq, casadi_int ny, casadi_int nl,
                    casadi_int nf, casadi_int rhs_cols)
        : euler_l(ny, nl), y_base(ny, rhs_cols), reduced_l(nl, nl),
          rnea_force(nq, nf), contact_schur(nf, nf),
          reduced(nl, rhs_cols), a_base(nq, rhs_cols),
          force(nf, rhs_cols), lifted(nl, rhs_cols),
          rhs(ny + nl, rhs_cols), output(ny + nl, rhs_cols) {}
    matrix euler_l;
    matrix y_base;
    matrix reduced_l;
    matrix rnea_force;
    matrix contact_schur;
    matrix reduced;
    matrix a_base;
    matrix force;
    matrix lifted;
    matrix rhs;
    matrix output;
    Eigen::PartialPivLU<matrix> orientation_factor;
    Eigen::PartialPivLU<matrix> gaa_factor;
    Eigen::PartialPivLU<matrix> contact_factor;
  } fused(nq, ny, nl, nf, rhs_cols);

  const auto run_fused = [&] {
    fused.rhs << hx_value, hu_value, hr_value;
    fused.orientation_factor.compute(orientation);
    const auto solve_euler = [&](const auto &input, matrix &output) {
      output.bottomRows(nq) = input.bottomRows(nq);
      output.topRows(nq).noalias() =
          input.topRows(nq) - coupling * input.bottomRows(nq);
      output.topRows(3).array().colwise() /= q_diag_head.array();
      output.middleRows(3, 3) = fused.orientation_factor.solve(
          output.middleRows(3, 3).eval());
      output.middleRows(6, 12).array().colwise() /= q_diag_tail.array();
    };
    solve_euler(dyn_l_value, fused.euler_l);
    fused.reduced_l.noalias() = lift_l_value - lift_y_value * fused.euler_l;
    fused.gaa_factor.compute(fused.reduced_l.topLeftCorner(nq, nq));
    fused.rnea_force = fused.gaa_factor.solve(
        fused.reduced_l.topRightCorner(nq, nf));
    fused.contact_schur.noalias() =
        fused.reduced_l.bottomRightCorner(nf, nf) -
        fused.reduced_l.bottomLeftCorner(nf, nq) * fused.rnea_force;
    fused.contact_schur.diagonal().array() += 1e-3;
    fused.contact_factor.compute(fused.contact_schur);
    solve_euler(fused.rhs.topRows(ny), fused.y_base);
    fused.reduced.noalias() =
        fused.rhs.bottomRows(nl) - lift_y_value * fused.y_base;
    fused.a_base = fused.gaa_factor.solve(fused.reduced.topRows(nq));
    fused.force = fused.contact_factor.solve(
        (fused.reduced.bottomRows(nf) -
         fused.reduced_l.bottomLeftCorner(nf, nq) * fused.a_base)
            .eval());
    fused.lifted.topRows(nq).noalias() =
        fused.a_base - fused.rnea_force * fused.force;
    fused.lifted.bottomRows(nf) = fused.force;
    fused.output.topRows(ny).noalias() =
        fused.y_base - fused.euler_l * fused.lifted;
    fused.output.bottomRows(nl) = fused.lifted;
  };

  graph(pointers);
  run_fused();
  REQUIRE(graph_output.isApprox(fused.output, 1e-10));

  if (std::getenv("MOTO_BENCH_LIFTED_ELIMINATION")) {
    constexpr size_t warmup = 200, iterations = 2000, rounds = 9;
    for (size_t i = 0; i < warmup; ++i) {
      graph(pointers);
      run_fused();
    }
    const auto measure = [&](auto &&function) {
      std::vector<double> samples;
      samples.reserve(rounds);
      for (size_t round = 0; round < rounds; ++round) {
        const auto begin = std::chrono::steady_clock::now();
        for (size_t i = 0; i < iterations; ++i) function();
        const auto end = std::chrono::steady_clock::now();
        samples.push_back(
            std::chrono::duration<double, std::micro>(end - begin).count() /
            iterations);
      }
      std::ranges::sort(samples);
      return samples[samples.size() / 2];
    };
    const double graph_us = measure([&] { graph(pointers); });
    const double fused_us = measure(run_fused);
    std::cerr << "Go2-sized lifted elimination: graph=" << graph_us
              << " us fused=" << fused_us
              << " us ratio=" << graph_us / fused_us << "x\n";
  }
}

TEST_CASE("MX graph entries reuse branch values and cached factors") {
  constexpr casadi_int n = 4, cols = 2;
  const casadi::MX a = casadi::MX::sym("entry_a", n, n);
  const casadi::MX c = casadi::MX::sym("entry_c", n, cols);
  const casadi::MX b = casadi::MX::sym("entry_b", n, cols);
  const casadi::MX shared = casadi::MX::mtimes(a, c);
  const casadi::MX first = casadi::MX::solve(a, shared);
  const casadi::MX second = casadi::MX::solve(a, shared + b);
  const auto kernel = compile_graph(
      {a, c, b}, std::vector<std::vector<casadi::MX>>{{first}, {second}},
      nullptr);

  matrix av = matrix::Random(n, n);
  av.diagonal().array() += 5.;
  matrix cv = matrix::Random(n, cols), bv = matrix::Random(n, cols);
  const matrix cached_a = av;
  const matrix cached_shared = av * cv;
  matrix first_value(n, cols), second_value(n, cols);
  std::vector<scalar_t *> pointers{av.data(), cv.data(), bv.data(),
                                   first_value.data(), second_value.data()};

  kernel(0, pointers);
  REQUIRE(first_value.isApprox(cv, 1e-12));

  av = matrix::Identity(n, n) * 17.;
  cv.setRandom();
  bv.setRandom();
  kernel(1, pointers);
  REQUIRE(second_value.isApprox(
      cached_a.partialPivLu().solve(cached_shared + bv), 1e-12));
}

TEST_CASE("MX graph presolve materializes branches shared only by actions") {
  constexpr casadi_int n = 4, cols = 2;
  const casadi::MX a = casadi::MX::sym("branch_a", n, n);
  const casadi::MX b = casadi::MX::sym("branch_b", n, cols);
  const casadi::MX c = casadi::MX::sym("branch_c", n, cols);
  const casadi::MX shared = casadi::MX::mtimes(a, b);
  const auto kernel = compile_graph(
      {a, b, c},
      std::vector<std::vector<casadi::MX>>{
          {a(casadi::Slice(0, 1), casadi::Slice(0, 1))},
          {casadi::MX::solve(a, shared)},
          {casadi::MX::solve(a, shared + c)}},
      nullptr);

  matrix av = matrix::Random(n, n);
  av.diagonal().array() += 5.;
  matrix bv = matrix::Random(n, cols), cv = matrix::Random(n, cols);
  scalar_t presolve_output = 0.;
  matrix first(n, cols), second(n, cols);
  std::vector<scalar_t *> pointers{av.data(), bv.data(), cv.data(),
                                   &presolve_output, first.data(),
                                   second.data()};

  kernel(0, pointers);
  kernel(2, pointers);
  REQUIRE(second.isApprox(
      av.partialPivLu().solve(av * bv + cv), 1e-12));
}

TEST_CASE("MX graph statically lowers a general sparse product") {
  const casadi::Sparsity a_pattern = casadi::Sparsity::triplet(
      4, 5, std::vector<casadi_int>{0, 2, 3, 1, 3},
      std::vector<casadi_int>{0, 0, 1, 3, 4});
  const casadi::Sparsity b_pattern = casadi::Sparsity::triplet(
      5, 3, std::vector<casadi_int>{0, 3, 1, 4},
      std::vector<casadi_int>{0, 0, 1, 2});
  const casadi::MX a = casadi::MX::sym("a", a_pattern);
  const casadi::MX b = casadi::MX::sym("b", b_pattern);
  const casadi::MX product = casadi::MX::mtimes(a, b);
  const auto kernel = compile_graph({a, b}, {product});

  vector av = vector::Random(a_pattern.nnz());
  vector bv = vector::Random(b_pattern.nnz());
  vector output(product.nnz());
  matrix a_dense = matrix::Zero(4, 5), b_dense = matrix::Zero(5, 3);
  for (casadi_int col = 0; col < a_pattern.size2(); ++col)
    for (casadi_int nz = a_pattern.colind(col);
         nz < a_pattern.colind(col + 1); ++nz)
      a_dense(a_pattern.row(nz), col) = av[nz];
  for (casadi_int col = 0; col < b_pattern.size2(); ++col)
    for (casadi_int nz = b_pattern.colind(col);
         nz < b_pattern.colind(col + 1); ++nz)
      b_dense(b_pattern.row(nz), col) = bv[nz];
  std::vector<scalar_t *> pointers{av.data(), bv.data(), output.data()};
  kernel(pointers);
  const matrix expected_dense = a_dense * b_dense;
  vector expected(product.nnz());
  const auto output_pattern = product.sparsity();
  for (casadi_int col = 0; col < output_pattern.size2(); ++col)
    for (casadi_int nz = output_pattern.colind(col);
         nz < output_pattern.colind(col + 1); ++nz)
      expected[nz] = expected_dense(output_pattern.row(nz), col);
  REQUIRE(output.isApprox(expected, 1e-12));
}

TEST_CASE("MX graph rejects unsupported ordinary operations") {
  const casadi::MX input = casadi::MX::sym("input", 3, 1);
  REQUIRE_THROWS(compile_graph({input}, {casadi::MX::sin(input)}));
}

} // namespace moto::linear_backend
