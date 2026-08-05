#include <catch2/catch_test_macros.hpp>

#include <moto/core/linear_backend.hpp>
#include <moto/core/sparse_matrix.hpp>

#include <casadi/casadi.hpp>
#include <cstdint>

namespace moto::linear_backend {
namespace {

struct fixture {
  condensation_spec spec;
  std::vector<matrix> jac;
  std::vector<vector> residual;
  std::vector<vector> weight;
  std::vector<row_vector> gradient;
  std::vector<matrix> hessian;

  explicit fixture(condensation_spec input) : spec(std::move(input)) {
    for (const size_t cols : spec.jac_cols) {
      jac.push_back(matrix::Random(spec.rows, cols));
      gradient.push_back(row_vector::Random(cols));
    }
    for (size_t side = 0; side < spec.residual_signs.size(); ++side) {
      residual.push_back(vector::Random(spec.rows));
      weight.push_back(vector::Random(spec.rows).cwiseAbs());
    }
    for (size_t i = 0; i < spec.jac_cols.size(); ++i)
      for (size_t j = i; j < spec.jac_cols.size(); ++j)
        hessian.push_back(matrix::Random(spec.jac_cols[i], spec.jac_cols[j]));
  }
};

void check_condensation(condensation_spec spec) {
  fixture f(std::move(spec));
  auto expected_gradient = f.gradient;
  auto expected_hessian = f.hessian;
  vector combined_residual = vector::Zero(f.spec.rows);
  vector combined_weight = vector::Zero(f.spec.rows);
  for (size_t side = 0; side < f.spec.residual_signs.size(); ++side) {
    combined_residual.noalias() +=
        f.spec.residual_signs[side] * f.residual[side];
    combined_weight.noalias() += f.weight[side];
  }
  for (size_t i = 0; i < f.jac.size(); ++i)
    expected_gradient[i].noalias() += combined_residual.transpose() * f.jac[i];
  size_t pair = 0;
  for (size_t i = 0; i < f.jac.size(); ++i)
    for (size_t j = i; j < f.jac.size(); ++j)
      expected_hessian[pair++].noalias() +=
          f.jac[i].transpose() * combined_weight.asDiagonal() * f.jac[j];

  auto kernel = compile_condensation(f.spec);
  std::vector<scalar_t *> pointers(kernel.pointer_count());
  for (size_t i = 0; i < f.jac.size(); ++i) {
    pointers[kernel.jacobian_slot(i)] = f.jac[i].data();
    pointers[kernel.gradient_slot(i)] = f.gradient[i].data();
  }
  for (size_t side = 0; side < f.residual.size(); ++side) {
    pointers[kernel.residual_slot(side)] = f.residual[side].data();
    pointers[kernel.weight_slot(side)] = f.weight[side].data();
  }
  pair = 0;
  for (size_t i = 0; i < f.jac.size(); ++i)
    for (size_t j = i; j < f.jac.size(); ++j)
      pointers[kernel.hessian_slot(i, j)] = f.hessian[pair++].data();
  kernel(pointers);

  for (size_t i = 0; i < f.gradient.size(); ++i)
    REQUIRE(f.gradient[i].isApprox(expected_gradient[i], 1e-12));
  for (size_t i = 0; i < f.hessian.size(); ++i)
    REQUIRE(f.hessian[i].isApprox(expected_hessian[i], 1e-12));
}

} // namespace

TEST_CASE("JIT condensation fuses sides and matches Eigen") {
  check_condensation(
      {.rows = 7, .jac_cols = {4, 3}, .residual_signs = {1., -1., .5}});
  check_condensation(
      {.rows = 24, .jac_cols = {18, 12}, .residual_signs = {1., -1.}});
}

TEST_CASE("JIT condensation source has no structural dispatch") {
  condensation_spec spec{
      .rows = 4, .jac_cols = {3, 2}, .residual_signs = {1., -1.}};
  const auto source = emit_condensation_source(spec);
  REQUIRE(source.find("switch") == std::string::npos);
  REQUIRE(source.find("if (") == std::string::npos);
  REQUIRE(source.find("moto_linear_axpy") != std::string::npos);
  REQUIRE(source.find("#include <Eigen") == std::string::npos);
}

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

TEST_CASE("CasADi-profiled SpGEMM matches a quadruped-sized product") {
  constexpr casadi_int n = 36, m = 30;
  std::vector<casadi_int> pr, pc, fr, fc;
  const auto add = [](auto &rows, auto &cols, casadi_int r0, casadi_int c0,
                      casadi_int nr, casadi_int nc) {
    for (casadi_int c = 0; c < nc; ++c)
      for (casadi_int r = 0; r < nr; ++r) {
        rows.push_back(r0 + r);
        cols.push_back(c0 + c);
      }
  };
  for (casadi_int i = 0; i < n; ++i) {
    pr.push_back(i);
    pc.push_back(i);
  }
  for (casadi_int i = 0; i < 18; ++i) {
    pr.push_back(i);
    pc.push_back(18 + i);
  }
  add(pr, pc, 3, 3, 3, 3);
  add(pr, pc, 3, 21, 3, 3);
  for (casadi_int i = 0; i < 18; ++i) {
    fr.push_back(i);
    fc.push_back(i);
  }
  add(fr, fc, 18, 0, 18, m);
  const auto psp = casadi::Sparsity::triplet(n, n, pr, pc);
  const auto fsp = casadi::Sparsity::triplet(n, m, fr, fc);
  auto kernel = compile_spgemm(psp, fsp);
  REQUIRE(kernel.output_layout().nnz() ==
          static_cast<size_t>(casadi::Sparsity::mtimes(psp, fsp).nnz()));
  REQUIRE_FALSE(kernel.output_layout().row_blocks.empty());
  REQUIRE_FALSE(kernel.output_layout().col_blocks.empty());

  vector pv = vector::Random(psp.nnz()), fv = vector::Random(fsp.nnz());
  vector out(kernel.output_layout().nnz());
  kernel(pv.data(), fv.data(), out.data());
  const auto dense = [](const auto &sp, const vector &values) {
    matrix result = matrix::Zero(sp.size1(), sp.size2());
    for (casadi_int c = 0; c < sp.size2(); ++c)
      for (casadi_int k = sp.colind(c); k < sp.colind(c + 1); ++k)
        result(sp.row(k), c) = values[k];
    return result;
  };
  matrix actual = matrix::Zero(n, m);
  const auto &layout = kernel.output_layout();
  for (size_t c = 0; c < layout.cols; ++c)
    for (size_t k = layout.colind[c]; k < layout.colind[c + 1]; ++k)
      actual(layout.row[k], c) = out[k];
  REQUIRE(actual.isApprox(dense(psp, pv) * dense(fsp, fv), 1e-12));
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

  matrix middle = matrix::Random(6, 6);
  matrix gram = matrix::Zero(5, 5);
  weighted_gram(sparse, middle, gram);
  REQUIRE(gram.isApprox(dense.transpose() * middle * dense, 1e-12));

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

  matrix weight = matrix::Random(6, 6), gram = matrix::Zero(6, 6);
  weighted_gram(sparse, weight, gram);
  REQUIRE(gram.isApprox(expected.transpose() * weight * expected, 1e-12));
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
  const product_spec spec{.sparse = describe(sparse),
                          .op = product_op::times,
                          .other_rows = 18,
                          .other_cols = 1,
                          .out_rows = 18,
                          .out_cols = 1};
  const auto source = emit_product_source(spec);
  REQUIRE(source.find("moto_linear_fused_diag_times") != std::string::npos);
  REQUIRE(source.find("#include <Eigen/Core>") == std::string::npos);
  REQUIRE(source.find("  moto_linear_diag_times(") == std::string::npos);
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
  REQUIRE(emit_product_source({.sparse = describe(sparse),
                               .op = product_op::times,
                               .other_rows = 6,
                               .other_cols = 5,
                               .out_rows = 8,
                               .out_cols = 5})
              .find("moto_linear_fused_dense_times") != std::string::npos);
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

} // namespace moto::linear_backend
