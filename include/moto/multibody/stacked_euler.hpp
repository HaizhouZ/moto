#ifndef MOTO_MULTIBODY_STACKED_EULER_HPP
#define MOTO_MULTIBODY_STACKED_EULER_HPP

#include <moto/multibody/impl/euler.hpp>
#include <moto/multibody/impl/euler_data.hpp>

namespace moto {
namespace multibody {

struct stacked_euler : public generic_dynamics {
    stacked_euler(const std::string &name) : generic_dynamics(name, approx_order::first, 0, __dyn) {}
    void add(euler &e) {
        dyn_.emplace_back(e);
        e.finalize();
    }
    void finalize_impl() override;
    struct approx_data : public generic_dynamics::approx_data {
        std::vector<euler_data> euler_data_;
        aligned_vector_map_t f_x_off_diag_, f_y_off_diag_, f_y_inv_off_diag_, proj_f_x_off_diag_;
        aligned_vector_map_t f_u_v_diag_, proj_f_u_v_diag_, proj_f_u_q_diag_;
        aligned_vector_map_t f_t, proj_f_t_;
        aligned_vector_map_t f_y_inv_main_diag_;
        sparse_mat f_y_inv;
        approx_data(generic_dynamics::approx_data &&rhs);
    };
    OVERLOAD_CREATE_APPROX_DATA(stacked_euler);
    void value_impl(func_approx_data &data) const override;
    void jacobian_impl(func_approx_data &data) const override;
    void compute_project_derivatives(func_approx_data &data) const override;
    void apply_jac_y_inverse_transpose(func_approx_data &data, vector &v, vector &dst) const override;
    bool wait_until_ready() const override;
    bool has_timestep_ = false;
    var dt_; // common timestep for all euler integrators
    size_t nq_ = 0, nv_ = 0;
    constexpr static size_t max_dyn = std::numeric_limits<size_t>::max();
    size_t exp_st_ = max_dyn, mid_st_ = max_dyn, imp_st_ = max_dyn;
    size_t dim_exp_ = 0, dim_mid_ = 0, dim_imp_ = 0;
    std::vector<func> dyn_;
};

} // namespace multibody

} // namespace moto

#endif