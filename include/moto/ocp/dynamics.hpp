#ifndef MOTO_OCP_DYNAMICS_HPP
#define MOTO_OCP_DYNAMICS_HPP

#include <moto/ocp/constr.hpp>
#include <moto/spmm/sparse_mat.hpp>

namespace moto {

/// @brief generic dynamics
struct generic_dynamics : public generic_constr {
    using base = generic_constr;
    struct approx_data : public base::approx_data {
        vector_ref proj_f_res_; ///< projection of f_res
        approx_data(base::approx_data &&rhs);
    };
    using base::base;
    virtual void compute_project_derivatives(func_approx_data &data) const = 0;
};

} // namespace moto

#endif // MOTO_OCP_DYNAMICS_HPP