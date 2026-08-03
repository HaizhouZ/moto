#ifndef MOTO_OCP_SPARSE_DYNAMICS_HPP
#define MOTO_OCP_SPARSE_DYNAMICS_HPP

#include <moto/core/linear_backend.hpp>
#include <moto/ocp/dynamics/dense_dynamics.hpp>

namespace moto {

/// Implicit dynamics using a numerically detected, statically compiled block
/// solve for the next-state Jacobian.
class sparse_dynamics : public dense_dynamics {
public:
  using base = dense_dynamics;
  using base::base;

  struct approx_data : public dense_dynamics::approx_data {
    linear_backend::multi_solve_kernel jac_solve_;
    linear_backend::multi_solve_kernel residual_solve_, transpose_solve_;
    std::vector<scalar_t *> jac_pointers_, residual_pointers_, transpose_pointers_;
    approx_data(generic_constr::approx_data &&rhs);
  };

  func_approx_data_ptr_t create_approx_data(sym_data &primal, lag_data &raw,
                                             shared_data &shared) const override {
    return func_approx_data_ptr_t(make_approx<sparse_dynamics>(primal, raw, shared));
  }
  void compute_project_jacobians(func_approx_data &data) const override;
  void compute_project_residual(func_approx_data &data) const override;
  void apply_jac_y_inverse_transpose(func_approx_data &data, vector &v,
                                     vector &dst) const override;

protected:
  clone_ptr clone() const override { return new sparse_dynamics(*this); }
  void finalize_impl() override;

private:
  linear_backend::solve_profile profile_, transpose_profile_;
  void analyze_profile();
};

} // namespace moto
#endif
