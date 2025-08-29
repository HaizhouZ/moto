#ifndef MOTO_MULTIBODY_EULER_HPP
#define MOTO_MULTIBODY_EULER_HPP

#include <moto/ocp/dynamics.hpp>

namespace moto {
namespace multibody {
struct euler_data;
struct stacked_euler;
// Euler angles utilities
class euler : public generic_func {
  private:
    using base = generic_func;
    friend struct stacked_euler;
    var q_, v_, a_, dt_;
    cs::SX pos_integrate_, pos_diff_, pos_step_;
    struct jac_block {
        bool empty = true;
        cs::SX dense, diag;
        struct {
            int r_st, c_st, nrow, ncol;
        } dense_block;
        bool has_block = false;
        bool has_diag = false; /// @note assumption diag exists if has_block is true
        void setup(const cs::SX &mat, size_t c_offset = 0);
    };
    struct {
        jac_block dqn, dq, dvn, dv;
    } pos_diff_jac_;

    struct {
        jac_block dq, dv, da;
    } pos_diff_proj_jac_;

    using binary_jac_t = std::array<cs::SX, 2>;
    binary_jac_t dpos_diff_, dpos_int_;

    enum class v_int_type : size_t {
        __explicit = 0,
        __mid_point,
        __implicit,
    };

    v_int_type v_int_type_ = v_int_type::__explicit;

  public:
    euler(const std::string &name,
          const var &q, const var &v, const var &a, const var &dt, cs::SX pos_step,
          cs::SX pos_diff, cs::SX dpos_diff, cs::SX pos_int, cs::SX dpos_int);
    void load_external_impl(const std::string &path) override;
    void finalize_impl() override;
    void setup_data(euler_data &data) const;
    void jacobian_impl(func_approx_data &data) const override;
    void compute_project_derivatives(euler_data &data) const;
};

struct euler_data : public func_approx_data {
    using so3_mat_t = Eigen::Matrix<scalar_t, 3, 3>;
    using so3_mat_map_t = null_init_map<so3_mat_t, false>;
    so3_mat_map_t f_y_inv_q_block, f_y_inv_v_block;
    so3_mat_map_t f_y_q_block, f_y_v_block, f_x_q_block, f_x_v_block;
    so3_mat_map_t proj_f_x_q_block, proj_f_x_v_block, proj_f_u_q_block;
    null_init_map<vector, false> f_u_v_diag_, f_t_q, f_t_v, f_y_v_off_diag_, f_x_v_off_diag_;
    null_init_map<vector, false> proj_f_t_q;
    using func_approx_data::func_approx_data;
};

struct stacked_euler : public generic_dynamics {
    void add(euler &e) {
        dyn_.emplace_back(e);
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
    void value_impl(func_approx_data &data) const override;
    void jacobian_impl(func_approx_data &data) const override;
    void compute_project_derivatives(func_approx_data &data) const override;
    void apply_jac_y_inverse_transpose(func_approx_data &data, vector& v, vector& dst) const override;
    bool has_timestep_ = false;
    var dt_; // common timestep for all euler integrators
    size_t nq_, nv_;
    size_t exp_st_, mid_st, imp_st_;
    constexpr static size_t max_dyn = std::numeric_limits<size_t>::max();
    size_t dim_exp_ = max_dyn, dim_mid_ = max_dyn, dim_imp_ = max_dyn;
    std::vector<func> dyn_;
};

} // namespace multibody
} // namespace moto

#endif // MOTO_MULTIBODY_EULER_HPP