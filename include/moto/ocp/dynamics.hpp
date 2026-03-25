#ifndef MOTO_OCP_DYNAMICS_HPP
#define MOTO_OCP_DYNAMICS_HPP

#include <moto/ocp/constr.hpp>
#include <moto/spmm/sparse_mat.hpp>

namespace moto {
class generic_dynamics;                           ///< forward declaration
using dynamics = utils::shared<generic_dynamics>; ///< shared pointer type for generic_dynamics
/// @brief generic dynamics
class generic_dynamics : public generic_constr {
  public:
    using base = generic_constr;
    struct approx_data : public base::approx_data {
#define NULL_INIT_MAP(name) name(nullptr, 0, 0)
#define NULL_INIT_VECMAP(name) name(nullptr, 0)
        merit_data::approx_data *approx_; ///< pointer to the merit data approx of dynamics field
        merit_data::dynamics_data *dyn_proj_; ///< pointer to the dynamics projection data
        vector_ref proj_f_res_; ///< projection of f_res
        approx_data(base::approx_data &&rhs);
    };
    using base::base;
    virtual void compute_project_jacobians(func_approx_data &data) const = 0;
    virtual void compute_project_residual(func_approx_data &data) const = 0;
    virtual void compute_project_derivatives(func_approx_data &data) const {
      compute_project_jacobians(data);
      compute_project_residual(data);
    }
    virtual void apply_jac_y_inverse_transpose(func_approx_data &data, vector &v, vector &dst) const { dst = v; };
};

} // namespace moto

#endif // MOTO_OCP_DYNAMICS_HPP