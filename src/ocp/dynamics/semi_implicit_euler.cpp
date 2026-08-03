#include <moto/ocp/dynamics/semi_implicit_euler.hpp>

#include <moto/ocp/problem.hpp>
#include <moto/utils/codegen.hpp>

namespace moto {
namespace {

struct piece { sp_info block; cs::SX value; };

cs::SX inverse3(const cs::SX &a) {
  const cs::SX det =
      a(0, 0) * (a(1, 1) * a(2, 2) - a(1, 2) * a(2, 1)) -
      a(0, 1) * (a(1, 0) * a(2, 2) - a(1, 2) * a(2, 0)) +
      a(0, 2) * (a(1, 0) * a(2, 1) - a(1, 1) * a(2, 0));
  return cs::SX::vertcat({
             cs::SX::horzcat({a(1, 1) * a(2, 2) - a(1, 2) * a(2, 1),
                              a(0, 2) * a(2, 1) - a(0, 1) * a(2, 2),
                              a(0, 1) * a(1, 2) - a(0, 2) * a(1, 1)}),
             cs::SX::horzcat({a(1, 2) * a(2, 0) - a(1, 0) * a(2, 2),
                              a(0, 0) * a(2, 2) - a(0, 2) * a(2, 0),
                              a(0, 2) * a(1, 0) - a(0, 0) * a(1, 2)}),
             cs::SX::horzcat({a(1, 0) * a(2, 1) - a(1, 1) * a(2, 0),
                              a(0, 1) * a(2, 0) - a(0, 0) * a(2, 1),
                              a(0, 0) * a(1, 1) - a(0, 1) * a(1, 0)})}) /
         det;
}

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
      for (size_t i = 0; i < n; ++i)
        used[r0 + i + (c0 + i) * nr] = 1;
      cs::SX diagonal = cs::SX::diag(
          expr(cs::Slice(static_cast<casadi_int>(r0), static_cast<casadi_int>(r)),
               cs::Slice(static_cast<casadi_int>(c0), static_cast<casadi_int>(c))));
      const sparsity pattern = diagonal.is_one() ? sparsity::eye : sparsity::diag;
      out.push_back({{pattern, r0, c0, n, n}, std::move(diagonal)});
    }
  }
  return out;
}

} // namespace

semi_implicit_euler::approx_data::approx_data(generic_constr::approx_data &&rhs)
    : generic_dynamics::approx_data(std::move(rhs), true) {
  const auto &dyn = static_cast<const semi_implicit_euler &>(func_);
  auto argument_refs = jac_;
  jac_.clear();
  const auto &prob = *lag_data_->prob_;
  const size_t f_st = prob.get_expr_start(func_);
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
  }
  scratch_.reserve(dyn.projected_panels_.size());
  for (const auto &[argument, block] : dyn.projected_panels_) {
    const sym &arg = func_.in_args(argument);
    if (!prob.is_active(arg)) {
      scratch_.emplace_back(block.rows,
                            block.pattern == sparsity::dense ? block.cols : 1);
      jac_.emplace_back(scratch_.back());
    } else {
      auto &target = arg.field() == __x ? dyn_proj_->proj_f_x_
                                        : dyn_proj_->proj_f_u_;
      jac_.push_back(target.insert(
          f_st + block.row_offset,
          prob.get_expr_start_tangent(arg) + block.col_offset,
          block.rows, block.cols, block.pattern));
    }
  }
  inverse_.resize(func_.dim(), func_.dim());
  for (const sp_info &block : dyn.inverse_panels_)
    jac_.push_back(inverse_.insert(block.row_offset, block.col_offset,
                                  block.rows, block.cols, block.pattern));
  inverse_pointers_ = linear_backend::panel_pointers(inverse_);
  residual_ = linear_backend::compile_product(
      {linear_backend::describe(inverse_), linear_backend::product_op::times,
       func_.dim(), 1, func_.dim(), 1});
  transpose_ = linear_backend::compile_product(
      {linear_backend::describe(inverse_),
       linear_backend::product_op::transpose_times, func_.dim(), 1,
       func_.dim(), 1});
}

void semi_implicit_euler::compute_project_jacobians(func_approx_data &data) const {
  (void)data;
}

void semi_implicit_euler::compute_project_residual(func_approx_data &data) const {
  auto &d = data.as<approx_data>();
  d.proj_f_res_.setZero();
  d.residual_(d.inverse_pointers_, d.approx_->v_.data(), d.proj_f_res_.data());
}

void semi_implicit_euler::apply_jac_y_inverse_transpose(
    func_approx_data &data, vector &v, vector &dst) const {
  auto &d = data.as<approx_data>();
  dst.setZero();
  d.transpose_(d.inverse_pointers_, v.data(), dst.data());
}

cs::SX semi_implicit_euler::configuration_inverse(const cs::SX &fy) {
  cs::SX inverse = cs::SX::diag(1 / cs::SX::diag(fy));
  if (fy.rows() >= 3) {
    const casadi_int orientation = fy.rows() >= 6 ? 3 : 0;
    const cs::Slice so3(orientation, orientation + 3);
    inverse(so3, so3) = inverse3(fy(so3, so3));
  }
  return inverse;
}

cs::SX semi_implicit_euler::symbolic_inverse(const cs::SX &fy) const {
  if (fy.rows() != fy.columns() || fy.rows() % 2)
    throw std::runtime_error(
        fmt::format("semi-implicit dynamics {} requires paired q/v state", name()));
  const casadi_int n = fy.rows() / 2;
  const cs::Slice q(0, n), v(n, 2 * n);
  const cs::SX a_inv = configuration_inverse(fy(q, q));
  return cs::SX::sparsify(cs::SX::vertcat(
      {cs::SX::horzcat({a_inv, -cs::SX::mtimes(a_inv, fy(q, v))}),
       cs::SX::horzcat({cs::SX::zeros(n, n), cs::SX::eye(n)})}));
}

void semi_implicit_euler::prepare_dynamics_codegen() {
  auto *task = get_codegen_task();
  if (!task)
    throw std::runtime_error("Euler dynamics requires a CasADi expression");
  jac_panels_.clear();
  projected_panels_.clear();
  inverse_panels_.clear();
  projected_profiles_.clear();
  task->jac_outputs.clear();
  std::vector<cs::SX> fy_blocks;
  for (size_t i = 0; i < in_args_.size(); ++i) {
    const sym &arg = in_args_[i];
    if (!in_field(arg.field(), primal_fields)) continue;
    cs::SX jac = cs::SX::sparsify(
        utils::cs_codegen::tangent_jacobian(task->sx_output, arg));
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
  const cs::SX inverse = symbolic_inverse(fy);

  for (size_t i = 0; i < in_args_.size(); ++i) {
    const sym &arg = in_args_[i];
    if (arg.field() != __x && arg.field() != __u)
      continue;
    cs::SX jac = cs::SX::sparsify(
        utils::cs_codegen::tangent_jacobian(task->sx_output, arg));
    cs::SX projected = cs::SX::mtimes(inverse, jac);
    projected_profiles_.push_back(
        linear_backend::analyze_sparsity(projected.sparsity()));
    if (std::getenv("MOTO_DEBUG_DYNAMICS_PROFILE")) {
      const auto &profile = projected_profiles_.back();
      fmt::println("{} P*F_{}: {}x{}, nnz={}/{}, blocks={}", name(),
                   arg.name(), profile.rows, profile.cols, profile.nnz(),
                   profile.rows * profile.cols,
                   profile.row_blocks.empty() ? 0
                                              : profile.row_blocks.size() - 1);
    }
    for (auto &[block, value] : split_panels(projected)) {
      projected_panels_.push_back({i, block});
      task->jac_outputs.push_back(std::move(value));
    }
  }
  for (auto &[block, value] : split_panels(inverse)) {
    inverse_panels_.push_back(block);
    task->jac_outputs.push_back(std::move(value));
  }
}

} // namespace moto
