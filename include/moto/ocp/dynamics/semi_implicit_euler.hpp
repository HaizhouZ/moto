#ifndef MOTO_OCP_SEMI_IMPLICIT_EULER_HPP
#define MOTO_OCP_SEMI_IMPLICIT_EULER_HPP

#include <moto/core/linear_backend.hpp>
#include <moto/ocp/dynamics.hpp>

namespace moto {

/// Euler dynamics with position or position-velocity state structure.
class semi_implicit_euler : public generic_dynamics {
public:
  using base = generic_dynamics;
  enum class state_t { pos, pos_vel };
  semi_implicit_euler(const std::string &name, const cs::SX &out,
                      state_t state = state_t::pos_vel,
                      approx_order order = approx_order::first);

  struct jac_panel {
    size_t argument = 0;
    sp_info block;
  };

  struct approx_data : public generic_dynamics::approx_data {
    sparse_matrix inverse_;
    linear_backend::product_kernel residual_, transpose_;
    std::vector<scalar_t *> inverse_pointers_;
    std::vector<matrix> scratch_;
    approx_data(generic_constr::approx_data &&rhs);
  };

  func_approx_data_ptr_t create_approx_data(sym_data &primal, lag_data &raw,
                                             shared_data &shared) const override {
    return func_approx_data_ptr_t(make_approx<semi_implicit_euler>(primal, raw, shared));
  }
  void compute_project_jacobians(func_approx_data &data) const override;
  void compute_project_residual(func_approx_data &data) const override;
  void apply_jac_y_inverse_transpose(func_approx_data &data, vector_ref v,
                                     vector_ref dst) const override;
  const auto &projected_profiles() const { return projected_profiles_; }

protected:
  clone_ptr clone() const override { return new semi_implicit_euler(*this); }
  void prepare_dynamics_codegen() override;
  cs::SX symbolic_inverse(const cs::SX &fy) const;
  static cs::SX configuration_inverse(const cs::SX &fy);

private:
  state_t state_ = state_t::pos_vel;
  std::vector<jac_panel> jac_panels_;
  std::vector<jac_panel> projected_panels_;
  std::vector<sp_info> inverse_panels_;
  std::vector<linear_backend::ccs_layout> projected_profiles_;
};

} // namespace moto
#endif
