#ifndef MOTO_OCP_DYNAMICS_HPP
#define MOTO_OCP_DYNAMICS_HPP

#include <moto/ocp/constr.hpp>
#include <moto/spmm/sparse_mat.hpp>

namespace moto {

struct dynamics_data_base {
    size_t dim_x;      ///< dimension of x
    size_t dim_u;      ///< dimension of u
    vector proj_f_res; ///< projection of f_res
    virtual sparse_mat &proj_f_x() = 0;
    virtual sparse_mat &proj_f_u() = 0;
};

// generic dense dynamics
struct generic_dynamics : public generic_constr {
    using base = generic_constr;
    struct approx_data : public base::approx_data, public dynamics_data_base {
        approx_data(base::approx_data &&rhs)
            : base::approx_data(std::move(rhs)) {
            dim_x = func_.arg_dim(__x);
            dim_u = func_.arg_dim(__u);
            proj_f_res.resize(dim_x);
            proj_f_res.setZero();
        }
    };
    using base::base;
    virtual void compute_project_derivatives(func_approx_data &data) = 0;
};

} // namespace moto

#endif // MOTO_OCP_DYNAMICS_HPP