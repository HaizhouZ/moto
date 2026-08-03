#include <moto/ocp/dynamics/sparse_dynamics.hpp>

#include <moto/utils/codegen.hpp>

namespace moto {
namespace {

linear_backend::solve_profile dense_profile(size_t dimension) {
  return {.dimension = dimension,
          .order = {{0, dimension, {}}},
          .inverse_nonzeros = std::vector<unsigned char>(dimension * dimension, 1),
          .dense_fallback = true};
}

matrix dm_matrix(const cs::DM &value) {
  const auto elements = cs::DM::densify(value).get_elements();
  matrix result(value.rows(), value.columns());
  std::copy(elements.begin(), elements.end(), result.data());
  return result;
}

} // namespace

sparse_dynamics::approx_data::approx_data(generic_constr::approx_data &&rhs)
    : dense_dynamics::approx_data(std::move(rhs)) {
  const auto &dyn = static_cast<const sparse_dynamics &>(func_);
  std::vector<size_t> columns;
  jac_pointers_.push_back(f_y_.data());
  const auto add = [&](matrix_ref input, matrix_ref output) {
    if (!input.size())
      return;
    columns.push_back(input.cols());
    jac_pointers_.push_back(input.data());
    jac_pointers_.push_back(output.data());
  };
  add(f_x_, proj_f_x_);
  add(f_u_exclusive_, proj_f_u_exclusive_);
  for (size_t i = 0; i < f_u_shared_.size(); ++i)
    add(f_u_shared_[i], proj_f_u_shared_[i]);
  jac_solve_ = linear_backend::compile_multi_solve(dyn.profile_, std::move(columns));
  residual_solve_ = linear_backend::compile_multi_solve(dyn.profile_, {1});
  transpose_solve_ = linear_backend::compile_multi_solve(dyn.transpose_profile_, {1}, true);
  residual_pointers_ = {f_y_.data(), approx_->v_.data(), proj_f_res_.data()};
  transpose_pointers_ = {f_y_.data(), nullptr, nullptr};
}

void sparse_dynamics::compute_project_jacobians(func_approx_data &data) const {
  auto &d = data.as<approx_data>();
  d.jac_solve_(d.jac_pointers_);
}

void sparse_dynamics::compute_project_residual(func_approx_data &data) const {
  auto &d = data.as<approx_data>();
  d.residual_solve_(d.residual_pointers_);
}

void sparse_dynamics::apply_jac_y_inverse_transpose(func_approx_data &data,
                                                     vector &v,
                                                     vector &dst) const {
  auto &d = data.as<approx_data>();
  d.transpose_pointers_[1] = v.data();
  d.transpose_pointers_[2] = dst.data();
  d.transpose_solve_(d.transpose_pointers_);
}

void sparse_dynamics::analyze_profile() {
  const auto *task = get_codegen_task();
  if (!task)
    throw std::runtime_error("sparse dynamics requires a CasADi expression");
  std::vector<cs::SX> blocks;
  size_t dimension = 0;
  for (const sym &arg : in_args_) {
    if (arg.field() != __y)
      continue;
    blocks.push_back(utils::cs_codegen::tangent_jacobian(task->sx_output, arg));
    dimension += arg.tdim();
  }
  if (dimension != dim())
    throw std::runtime_error(fmt::format(
        "sparse dynamics {} requires square F_y, got {}x{}", name(), dim(), dimension));
  const cs::SX fy = cs::SX::horzcat(blocks);
  try {
    std::vector<cs::SX> sx_inputs;
    sx_inputs.reserve(in_args_.size());
    for (const sym &arg : in_args_)
      sx_inputs.emplace_back(arg);
    cs::Function evaluate(name() + "_fy_profile", sx_inputs, {fy});
    std::array<matrix, 3> samples;
    for (size_t sample = 0; sample < samples.size(); ++sample) {
      std::vector<cs::DM> inputs;
      for (const sym &arg : in_args_) {
        vector value = arg.default_value().size() == static_cast<Eigen::Index>(arg.dim())
                           ? arg.default_value()
                           : vector::Zero(arg.dim());
        if (sample && arg.has_non_trivial_integration()) {
          if (!arg.default_value().size())
            throw std::runtime_error("manifold profile sampling requires a default value");
          vector step(arg.tdim()), perturbed(arg.dim());
          for (size_t i = 0; i < arg.tdim(); ++i)
            step[i] = .03 * sample * std::sin(scalar_t(arg.uid() + 17 * i));
          arg.integrate(value, step, perturbed);
          value = std::move(perturbed);
        } else if (sample) {
          for (size_t i = 0; i < arg.dim(); ++i)
            value[i] += .03 * sample * std::sin(scalar_t(arg.uid() + 17 * i));
        }
        inputs.emplace_back(std::vector<scalar_t>(value.data(), value.data() + value.size()));
      }
      samples[sample] = dm_matrix(evaluate(inputs).front());
    }
    profile_ = linear_backend::analyze_solve_profile(samples);
    std::array<matrix, 3> transpose_samples;
    for (size_t i = 0; i < samples.size(); ++i)
      transpose_samples[i] = samples[i].transpose();
    transpose_profile_ = linear_backend::analyze_solve_profile(transpose_samples);
  } catch (const std::exception &error) {
    fmt::print(stderr, "sparse dynamics {} profile detection fell back to dense: {}\n",
               name(), error.what());
    profile_ = dense_profile(dimension);
    transpose_profile_ = profile_;
  }
}

void sparse_dynamics::finalize_impl() {
  analyze_profile();
  dense_dynamics::finalize_impl();
}

} // namespace moto
