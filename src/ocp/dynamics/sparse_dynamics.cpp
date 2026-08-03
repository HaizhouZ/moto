#include <moto/ocp/dynamics/sparse_dynamics.hpp>

#include <moto/ocp/problem.hpp>
#include <moto/utils/codegen.hpp>

namespace moto {
namespace {

struct piece { sp_info block; cs::SX value; };

std::vector<piece> split_panels(const cs::SX &input) {
  const cs::SX expr = cs::SX::sparsify(input);
  const size_t nr = expr.rows(), nc = expr.columns();
  std::vector<unsigned char> used(nr * nc);
  const auto live = [&](size_t r, size_t c) {
    return !used[r + c * nr] && !expr(r, c).is_zero();
  };
  const auto mark = [&](size_t r, size_t c, size_t rows, size_t cols) {
    for (size_t j = c; j < c + cols; ++j)
      for (size_t i = r; i < r + rows; ++i) used[i + j * nr] = 1;
  };
  std::vector<piece> out;
  // Extract non-trivial dense rectangles first (quaternion and ordinary dense blocks).
  for (size_t r = 0; r < nr; ++r)
    for (size_t c = 0; c < nc; ++c) {
      if (!live(r, c)) continue;
      size_t cols = 0;
      while (c + cols < nc && live(r, c + cols)) ++cols;
      size_t rows = 1;
      while (r + rows < nr) {
        bool full = true;
        for (size_t j = 0; j < cols; ++j) full &= live(r + rows, c + j);
        if (!full) break;
        ++rows;
      }
      if (rows < 2 || cols < 2) continue;
      mark(r, c, rows, cols);
      out.push_back({{sparsity::dense, r, c, rows, cols},
                     expr(cs::Slice(static_cast<casadi_int>(r), static_cast<casadi_int>(r + rows)),
                          cs::Slice(static_cast<casadi_int>(c), static_cast<casadi_int>(c + cols)))});
    }
  // Fuse every remaining shifted diagonal into the longest possible run.
  for (ptrdiff_t offset = -static_cast<ptrdiff_t>(nr) + 1;
       offset < static_cast<ptrdiff_t>(nc); ++offset) {
    size_t r = offset < 0 ? -offset : 0, c = offset > 0 ? offset : 0;
    while (r < nr && c < nc) {
      while (r < nr && c < nc && !live(r, c)) { ++r; ++c; }
      const size_t r0 = r, c0 = c;
      while (r < nr && c < nc && live(r, c)) { ++r; ++c; }
      if (r == r0) continue;
      const size_t n = r - r0;
      mark(r0, c0, n, n);
      cs::SX diagonal = cs::SX::diag(
          expr(cs::Slice(static_cast<casadi_int>(r0), static_cast<casadi_int>(r)),
               cs::Slice(static_cast<casadi_int>(c0), static_cast<casadi_int>(c))));
      const sparsity pattern = diagonal.is_one() ? sparsity::eye : sparsity::diag;
      out.push_back({{pattern, r0, c0, n, n}, std::move(diagonal)});
    }
  }
  return out;
}

matrix dm_matrix(const cs::DM &value) {
  const auto values = cs::DM::densify(value).get_elements();
  matrix result(value.rows(), value.columns());
  std::copy(values.begin(), values.end(), result.data());
  return result;
}

linear_backend::solve_profile dense_profile(size_t n) {
  return {.dimension = n,
          .order = {{0, n, {}}},
          .inverse_nonzeros = std::vector<unsigned char>(n * n, 1),
          .dense_fallback = true};
}

} // namespace

sparse_dynamics::approx_data::approx_data(generic_constr::approx_data &&rhs)
    : generic_dynamics::approx_data(std::move(rhs)) {
  const auto &dyn = static_cast<const sparse_dynamics &>(func_);
  auto argument_refs = jac_;
  jac_.clear();
  std::vector<scalar_t *> fy_pointers;
  linear_backend::matrix_layout fy_layout{func_.dim(), func_.arg_tdim(__y), {}};
  const auto &prob = *lag_data_->prob_;
  const size_t f_st = prob.get_expr_start(func_);
  std::vector<size_t> y_offsets(func_.in_args().size());
  size_t y_col = 0;
  for (size_t i = 0; i < func_.in_args().size(); ++i)
    if (func_.in_args(i)->field() == __y) {
      y_offsets[i] = y_col;
      y_col += func_.in_args(i)->tdim();
    }
  for (const auto &[argument, block] : dyn.jac_panels_) {
    const sym &arg = func_.in_args(argument);
    if (arg.field() != __y) {
      jac_.push_back(argument_refs[argument]);
      continue;
    }
    auto ref = approx_->jac_[__y].insert(
        f_st + block.row_offset,
        prob.get_expr_start_tangent(arg) + block.col_offset,
        block.rows, block.cols, block.pattern);
    jac_.push_back(ref);
    fy_pointers.push_back(ref.data());
    fy_layout.panels.push_back({block.pattern, block.row_offset,
                                y_offsets[argument] + block.col_offset,
                                block.rows, block.cols});
  }

  auto profile = dyn.profile_;
  profile.lhs = fy_layout;
  auto transpose_profile = dyn.transpose_profile_;
  transpose_profile.lhs = fy_layout;
  std::vector<size_t> columns;
  jac_pointers_ = fy_pointers;
  const auto add_rhs = [&](matrix_ref input, matrix_ref output) {
    if (!input.size()) return;
    columns.push_back(input.cols());
    jac_pointers_.push_back(input.data());
    jac_pointers_.push_back(output.data());
  };
  add_rhs(f_x_, proj_f_x_);
  add_rhs(f_u_exclusive_, proj_f_u_exclusive_);
  for (size_t i = 0; i < f_u_shared_.size(); ++i)
    add_rhs(f_u_shared_[i], proj_f_u_shared_[i]);
  jac_solve_ = linear_backend::compile_multi_solve(profile, std::move(columns));
  residual_solve_ = linear_backend::compile_multi_solve(profile, {1});
  transpose_solve_ = linear_backend::compile_multi_solve(transpose_profile, {1}, true);
  residual_pointers_ = fy_pointers;
  residual_pointers_.insert(residual_pointers_.end(),
                            {approx_->v_.data(), proj_f_res_.data()});
  transpose_pointers_ = fy_pointers;
  transpose_pointers_.insert(transpose_pointers_.end(), {nullptr, nullptr});
}

void sparse_dynamics::compute_project_jacobians(func_approx_data &data) const {
  data.as<approx_data>().jac_solve_(data.as<approx_data>().jac_pointers_);
}

void sparse_dynamics::compute_project_residual(func_approx_data &data) const {
  data.as<approx_data>().residual_solve_(data.as<approx_data>().residual_pointers_);
}

void sparse_dynamics::apply_jac_y_inverse_transpose(func_approx_data &data,
                                                     vector &v, vector &dst) const {
  auto &d = data.as<approx_data>();
  d.transpose_pointers_.end()[-2] = v.data();
  d.transpose_pointers_.back() = dst.data();
  d.transpose_solve_(d.transpose_pointers_);
}

void sparse_dynamics::prepare_dynamics_codegen() {
  auto *task = get_codegen_task();
  if (!task) throw std::runtime_error("sparse dynamics requires a CasADi expression");
  jac_panels_.clear();
  task->jac_outputs.clear();
  std::vector<cs::SX> fy_blocks, inputs;
  for (size_t i = 0; i < in_args_.size(); ++i) {
    const sym &arg = in_args_[i];
    inputs.emplace_back(arg);
    if (!in_field(arg.field(), primal_fields)) continue;
    cs::SX jac = utils::cs_codegen::tangent_jacobian(task->sx_output, arg);
    if (arg.field() == __y) {
      fy_blocks.push_back(jac);
      for (auto &[block, value] : split_panels(jac)) {
        jac_panels_.push_back({i, block});
        task->jac_outputs.push_back(std::move(value));
      }
    } else {
      jac_panels_.push_back({i, {sparsity::dense, 0, 0,
                                 static_cast<size_t>(jac.rows()),
                                 static_cast<size_t>(jac.columns())}});
      task->jac_outputs.push_back(std::move(jac));
    }
  }
  const cs::SX fy = cs::SX::horzcat(fy_blocks);
  if (fy.rows() != fy.columns())
    throw std::runtime_error(fmt::format("sparse dynamics {} has non-square F_y", name()));
  cs::Function evaluate(name() + "_fy_profile", inputs, {fy});
  std::array<matrix, 3> samples;
  for (size_t sample = 0; sample < samples.size(); ++sample) {
    std::vector<cs::DM> values;
    for (const sym &arg : in_args_) {
      vector value = arg.default_value().size() ? arg.default_value()
                                                : vector::Zero(arg.dim());
      if (sample && arg.has_non_trivial_integration()) {
        if (!arg.default_value().size())
          throw std::runtime_error("manifold profile sampling requires defaults");
        vector step(arg.tdim()), integrated(arg.dim());
        for (size_t i = 0; i < arg.tdim(); ++i)
          step[i] = .03 * sample * std::sin(scalar_t(arg.uid() + 17 * i));
        arg.integrate(value, step, integrated);
        value = std::move(integrated);
      } else if (sample) {
        for (size_t i = 0; i < arg.dim(); ++i)
          value[i] += .03 * sample * std::sin(scalar_t(arg.uid() + 17 * i));
      }
      values.emplace_back(std::vector<scalar_t>(value.data(), value.data() + value.size()));
    }
    samples[sample] = dm_matrix(evaluate(values).front());
  }
  try {
    profile_ = linear_backend::analyze_solve_profile(samples);
    std::array<matrix, 3> transposed;
    for (size_t i = 0; i < samples.size(); ++i)
      transposed[i] = samples[i].transpose();
    transpose_profile_ = linear_backend::analyze_solve_profile(transposed);
  } catch (const std::exception &error) {
    fmt::print(stderr, "sparse dynamics {} uses dense solve: {}\n", name(),
               error.what());
    profile_ = transpose_profile_ = dense_profile(fy.rows());
  }
}

} // namespace moto
