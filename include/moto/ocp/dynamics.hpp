#ifndef MOTO_OCP_DYNAMICS_HPP
#define MOTO_OCP_DYNAMICS_HPP

#include <moto/ocp/constr.hpp>
#include <moto/core/sparse_matrix.hpp>

namespace moto {
class generic_dynamics;                           ///< forward declaration
using dynamics = utils::shared<generic_dynamics>; ///< shared pointer type for generic_dynamics
/// @brief generic dynamics
class generic_dynamics : public generic_constr {
  public:
    using base = generic_constr;
    struct projection_panel {
      size_t argument = 0;
      sp_info block;
    };
    struct approx_data : public base::approx_data {
#define NULL_INIT_MAP(name) name(nullptr, 0, 0)
#define NULL_INIT_VECMAP(name) name(nullptr, 0)
        lag_data::approx_data *approx_;       ///< pointer to the lag data approx of dynamics field
        lag_data::dynamics_data *dyn_proj_;   ///< pointer to the dynamics projection data
        vector_ref proj_f_res_; ///< projection of f_res
        aligned_map_t f_x_;
        aligned_map_t f_u_exclusive_, proj_f_u_exclusive_;
        std::vector<aligned_map_t> f_u_shared_, proj_f_u_shared_;
        aligned_map_t proj_f_x_;
        approx_data(base::approx_data &&rhs);
        approx_data(base::approx_data &&rhs, bool sparse_projection);
    };
    using base::base;
    void mark_shared_inputs(const var_inarg_list &args);
    bool input_shared(const sym &s) const;
    virtual void compute_project_jacobians(func_approx_data &data) const = 0;
    virtual void compute_project_residual(func_approx_data &data) const = 0;
    virtual void compute_project_derivatives(func_approx_data &data) const {
      compute_project_jacobians(data);
      compute_project_residual(data);
    }
    virtual void apply_jac_y_inverse_transpose(func_approx_data &data,
                                               vector_ref v,
                                               vector_ref dst) const { dst = v; }
    virtual std::span<const projection_panel> projected_panel_sparsity() const {
      return {};
    }

  protected:
    var_list shared_inputs_;
    std::set<size_t> shared_inputs_indices_;
    void finalize_impl() override;
    void substitute(const sym &arg, const sym &rhs) override;
    virtual void prepare_dynamics_codegen() {}
};

} // namespace moto

#endif // MOTO_OCP_DYNAMICS_HPP
