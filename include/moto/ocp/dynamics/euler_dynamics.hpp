#ifndef MOTO_OCP_DYNAMICS_EULER_DYNAMICS_HPP
#define MOTO_OCP_DYNAMICS_EULER_DYNAMICS_HPP

#include <moto/ocp/dynamics.hpp>

namespace moto {
struct euler : public generic_func {
    struct state {
        var quat, w;
        var r, v;
        var dw, a, dt;
    };
    cs::SX d_so3_dquat_n, d_so3_dw_n, d_so3_dquat, d_so3_dw;
    struct euler_data {
        // to be computed by generated function
        aligned_map_t f_y_quat, f_x_quat, f_y_w, f_x_w;
        null_init_map<vector, false> f_x_v, f_y_v;  // because dq_step_d(v/vn) may be varying, e.g., v * dt or (v + vn) * dt / 2
        null_init_map<vector, false> f_t_quat, f_t_w, f_t_r, f_t_v;
        // this is from stacked_euler cuz it is just dt
        null_init_map<vector, false> f_u_dw, f_u_a; // dfda
        // to be computed by update_inv
        aligned_map_t f_y_quat_inv, f_y_inv_w;
        null_init_map<vector, false> f_y_inv_v, proj_f_x_v, proj_f_u_so3_a;
        aligned_map_t proj_f_y_quat, proj_f_y_w, proj_f_u_so3_dw;
    };
    bool has_so3_ = false, has_lin_ = false;
    void finalize_impl() override {
        // generate a function to compute the value
        // for the jacobian output only the necessary parts
        // also generate a function for so3 integration
    }
    void update_so3_inv(euler_data &d) const {
        d.f_y_quat_inv = d.f_y_quat.inverse();
        bool has_y_w = false;
        if (has_y_w = (d.f_y_inv_w.size() > 0))
            d.f_y_inv_w.noalias() = -d.f_y_quat_inv * d.f_y_w;
        d.proj_f_y_quat.noalias() = d.f_y_quat_inv * d.f_x_quat;
        d.proj_f_y_w.setZero();
        if (has_y_w)
            d.proj_f_y_w -= d.f_y_inv_w; // -1.0 * d.f_y_inv_w
        bool has_x_w = false;
        if (has_x_w = (d.f_x_w.size() > 0))
            d.proj_f_y_w.noalias() += d.f_y_quat_inv * d.f_x_w;
        if (has_y_w)
            d.proj_f_u_so3_dw.noalias() = d.f_y_inv_w * d.f_u_dw.asDiagonal();
    }
    void update_lin_inv(euler_data &d) const {
        bool has_y_v = false;
        if (has_y_v = (d.f_y_inv_v.size() > 0))
            d.f_y_inv_v = -d.f_y_v;
        d.proj_f_x_v.setZero();
        if (has_y_v)
            d.proj_f_x_v -= d.f_y_inv_v; // -1.0 * d.f_y_inv_v
        bool has_x_v = false;
        if (has_x_v = (d.f_x_v.size() > 0))
            d.proj_f_x_v += d.f_x_v;
        if (has_y_v)
            d.proj_f_u_so3_a.noalias() = d.f_y_inv_v.cwiseProduct(d.f_u_a);
    }
    /// @brief update the inverse jacobian related terms
    /// @param d data to update
    /// this function does not update proj_f_u (for dw and a) and proj_f_t
    void update_inv(euler_data &d) const {
        if (has_so3_)
            update_so3_inv(d);
        if (has_lin_)
            update_lin_inv(d);
    }
};

struct stacked_euler : public generic_dynamics {
    std::vector<euler *> dyns_;
    var dt_;
    struct approx_data : public generic_dynamics::approx_data {
        aligned_vector_map_t f_u, proj_f_u; // dfda
        aligned_vector_map_t f_t, proj_f_t; // dfdt
        sparse_mat f_y_inv;
        std::vector<euler::euler_data> dyn_data;
        approx_data(base::approx_data &&rhs) : generic_dynamics::approx_data(std::move(rhs)) {}
    };
    void value_impl(func_approx_data &data) const override {
        for (auto &dyn : dyns_) {
            dyn->value(data); // call each dynamics value
        }
    }
    void jacobian_impl(func_approx_data &data) const override {
        auto &d = data.as<approx_data>();
        scalar_t dt = data[dt_](0);
        d.f_u.setConstant(-dt); // for dw and a
        for (auto &dyn : dyns_) {
            dyn->jacobian(data); // call each dynamics jacobian
        }
    }
    void compute_project_jacobians(func_approx_data &data) const override {
        auto &d = data.as<approx_data>();
        size_t idx = 0;
        for (auto &dyn : dyns_) {
            dyn->update_inv(d.dyn_data[idx++]); // call each dynamics project derivatives
        }
        if (dt_) {
            d.proj_f_t.setZero();
            d.f_y_inv.times(d.f_t, d.proj_f_t);
        }
    }
    void compute_project_residual(func_approx_data &data) const override {
        auto &d = data.as<approx_data>();
        d.f_y_inv.times(d.v_, d.proj_f_res_);
    }
};
} // namespace moto

#endif