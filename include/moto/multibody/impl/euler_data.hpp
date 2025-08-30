#ifndef MOTO_MULTIBODY_EULER_DATA_HPP
#define MOTO_MULTIBODY_EULER_DATA_HPP

#include <moto/ocp/impl/func_data.hpp>

namespace moto {

namespace multibody {
struct euler_data : public func_approx_data {
    using so3_mat_t = Eigen::Matrix<scalar_t, 3, 3>;
    using so3_mat_map_t = null_init_map<so3_mat_t>;
    so3_mat_map_t f_y_inv_q_block, f_y_inv_v_block;
    so3_mat_map_t f_y_q_block, f_y_v_block, f_x_q_block, f_x_v_block;
    so3_mat_map_t proj_f_x_q_block, proj_f_x_v_block, proj_f_u_q_block;
    null_init_map<vector, false> f_u_v_diag_, f_t_q, f_t_v, f_y_v_off_diag_, f_x_v_off_diag_;
    null_init_map<vector, false> proj_f_t_q;
    using func_approx_data::func_approx_data;
};

} // namespace multibody
} // namespace moto

#endif