#include <moto/multibody/stacked_euler.hpp>
#include <moto/ocp/dynamics.hpp>
#include <moto/ocp/problem.hpp>

namespace moto {
namespace multibody {

bool stacked_euler::wait_until_ready() const {
    for (const euler &e : dyn_) {
        e.wait_until_ready();
    }
    return generic_dynamics::wait_until_ready();
}

void stacked_euler::finalize_impl() {
    // make sure the arg order is [pos, vel] !
    std::sort(dyn_.begin(), dyn_.end(), [](const euler &a, const euler &b) {
        return a.v_int_type_ <= b.v_int_type_;
    });
    dt_ = ((euler &)dyn_[0]).dt;
    has_timestep_ = dt_->field() == __u;
    for (const euler &e : dyn_) {
        if (dt_ != e.dt) {
            throw std::runtime_error("all euler dynamics must have the same time step variable");
        }
        if (e.v_int_type_ == euler::v_int_type::_implicit) {
            imp_st_ = std::min(imp_st_, nv_);
            dim_imp_ += e.v->dim();
        } else if (e.v_int_type_ == euler::v_int_type::_mid_point) {
            mid_st_ = std::min(mid_st_, nv_);
            dim_mid_ += e.v->dim();
        } else if (e.v_int_type_ == euler::v_int_type::_explicit) {
            exp_st_ = std::min(exp_st_, nv_);
            dim_exp_ += e.v->dim();
        }
        add_argument(e.q);
        add_argument(e.q->next());
        add_argument(e.a);
        nq_ += e.q->tdim();
        nv_ += e.v->dim();
        dim_ += e.dim();
    }
    for (euler &e : dyn_) {
        add_argument(e.v);
        add_argument(e.v->next());
    }
    add_argument(dt_);
    generic_dynamics::finalize_impl();
    // for (const auto &arg : in_args_) {
    //     fmt::println("arg: {} of uid {} in field {} with dim {}", arg->name(), arg->uid(), arg->field(), arg->dim());
    // }
}
stacked_euler::approx_data::approx_data(generic_dynamics::approx_data &&rhs) : generic_dynamics::approx_data(std::move(rhs)) {
    const auto &dyn = static_cast<const stacked_euler &>(func_);
    const auto &prob = *problem();
    size_t f_st = prob.get_expr_start(func_);
    size_t y_st = prob.get_expr_start_tangent(dyn.in_args(__y)[0]);
    size_t x_st = prob.get_expr_start_tangent(dyn.in_args(__x)[0]);
    size_t u_st = prob.get_expr_start_tangent(dyn.in_args(__u)[0]);
    size_t t_st = 0;
    // setup main diag
    vector_ref f_y_main_diag = approx_->jac_[__y].insert<sparsity::diag>(f_st, y_st, func_.dim()).setConstant(1);
    vector_ref f_x_main_diag = approx_->jac_[__x].insert<sparsity::diag>(f_st, x_st, func_.dim()).setConstant(-1);
    vector_ref proj_f_x_main_diag = dyn_proj_->proj_f_x_.insert<sparsity::diag>(f_st, x_st, func_.dim()).setConstant(-1);
    approx_->jac_[__u].insert<sparsity::diag>(f_st + dyn.nv_, u_st, dyn.nv_);
    // setup inverse
    f_y_inv.resize(func_.dim(), func_.dim());
    vector_ref f_y_inv_main_diag = f_y_inv.insert<sparsity::diag>(f_st, y_st, func_.dim()).setConstant(1);
    setup_map(f_y_inv_main_diag_, f_y_inv_main_diag);
    // setup off-diag
    size_t x_off_diag_rst = 0;
    size_t x_off_diag_st = dyn.exp_st_ < max_dyn ? dyn.exp_st_ : dyn.mid_st_;
    vector_ref off_diag_x = approx_->jac_[__x].insert<sparsity::diag>(f_st + x_off_diag_rst, x_st + dyn.nv_ + x_off_diag_st, dyn.dim_exp_ + dyn.dim_mid_);
    setup_map(f_x_off_diag_, off_diag_x);
    vector_ref proj_off_diag_x = dyn_proj_->proj_f_x_.insert<sparsity::diag>(f_st, x_st + dyn.nv_, dyn.nv_);
    setup_map(proj_f_x_off_diag_, proj_off_diag_x);
    size_t y_off_diag_rst = dyn.mid_st_ < max_dyn ? dyn.mid_st_ : dyn.imp_st_;
    size_t y_off_diag_st = dyn.mid_st_ < max_dyn ? dyn.mid_st_ : dyn.imp_st_;
    vector_ref off_diag_y = approx_->jac_[__y].insert<sparsity::diag>(f_st + y_off_diag_rst, x_st + dyn.nv_ + y_off_diag_st, dyn.dim_imp_ + dyn.dim_mid_);
    setup_map(f_y_off_diag_, off_diag_y);
    vector_ref off_diag_y_inv = f_y_inv.insert<sparsity::diag>(f_st + y_off_diag_rst, x_st + dyn.nv_ + y_off_diag_st, dyn.dim_imp_ + dyn.dim_mid_);
    setup_map(f_y_inv_off_diag_, off_diag_y_inv);
    vector_ref u_v_diag = approx_->jac_[__u].insert<sparsity::diag>(f_st + dyn.nv_, u_st, dyn.nv_);
    setup_map(f_u_v_diag_, u_v_diag);
    // setup the projection of input jacobian diag
    vector_ref proj_u_diag_q = dyn_proj_->proj_f_u_.insert<sparsity::diag>(f_st + y_off_diag_rst, u_st + y_off_diag_st, off_diag_y.size()); /// @todo wrong length
    setup_map(proj_f_u_q_diag_, proj_u_diag_q);
    vector_ref proj_u_diag_v = dyn_proj_->proj_f_u_.insert<sparsity::diag>(f_st + dyn.nv_, u_st, dyn.nv_);
    setup_map(proj_f_u_v_diag_, proj_u_diag_v);
    if (dyn.has_timestep_) {
        t_st = prob.get_expr_start_tangent(dyn.dt_);
        auto ft = approx_->jac_[__u].insert(f_st, t_st, dyn.dim(), 1, sparsity::dense);
        setup_map(f_t, ft);
        auto proj_ft = dyn_proj_->proj_f_u_.insert(f_st, t_st, dyn.dim(), 1, sparsity::dense);
        setup_map(proj_f_t_, proj_ft);
    }
    size_t e_st = 0;
    size_t f_x_off_diag_st_offset = x_off_diag_st;
    size_t f_y_off_diag_st_offset = y_off_diag_st;
    for (euler &e : dyn.dyn_) {
        size_t q_st = prob.get_expr_start_tangent(e.q);
        size_t v_st = prob.get_expr_start_tangent(e.v);
        size_t qn_st = prob.get_expr_start_tangent(e.q->next());
        size_t vn_st = prob.get_expr_start_tangent(e.v->next());
        size_t a_st = prob.get_expr_start_tangent(e.a);
        euler_data d(*this->primal_, *this->merit_data_, this->shared_, func_);
        auto set_sp_mat = [&, this](sparse_mat &dst, euler::jac_block &jb, size_t c_st, auto &_map, bool is_vel = false) {
            auto m = dst.insert(f_st + e_st + jb.dense_block.r_st,
                                c_st + jb.dense_block.c_st,
                                jb.dense_block.nrow,
                                jb.dense_block.ncol, sparsity::dense);
            setup_map(_map, m);
        };
        auto set_diag_zero = [&, this](auto &diag_mat, euler::jac_block &jb, size_t st_offset = 0) {
            if (diag_mat.size() == 0)
                return;
            diag_mat.segment(e_st - st_offset + jb.dense_block.r_st, jb.dense_block.nrow).setZero();
        };
        if (e.pos_diff_jac_.dqn.has_block) {
            set_sp_mat(approx_->jac_[__y], e.pos_diff_jac_.dqn, qn_st, d.f_y_q_block);
            set_sp_mat(f_y_inv, e.pos_diff_jac_.dqn, qn_st, d.f_y_inv_q_block);
            set_diag_zero(f_y_main_diag, e.pos_diff_jac_.dqn);
            set_diag_zero(f_y_inv_main_diag, e.pos_diff_jac_.dqn);
        }
        if (e.pos_diff_proj_jac_.dq.has_block) {
            set_sp_mat(dyn_proj_->proj_f_x_, e.pos_diff_proj_jac_.dq, q_st, d.proj_f_x_q_block);
            set_diag_zero(proj_f_x_main_diag, e.pos_diff_proj_jac_.dq);
        }
        if (e.pos_diff_jac_.dq.has_block) {
            set_sp_mat(approx_->jac_[__x], e.pos_diff_jac_.dq, q_st, d.f_x_q_block);
            set_diag_zero(f_x_main_diag, e.pos_diff_jac_.dq);
        }
        if (e.pos_diff_jac_.dvn.has_block) {
            set_sp_mat(approx_->jac_[__y], e.pos_diff_jac_.dvn, vn_st, d.f_y_v_block, true);
            set_sp_mat(f_y_inv, e.pos_diff_jac_.dvn, v_st, d.f_y_inv_v_block, true);
            set_diag_zero(f_y_off_diag_, e.pos_diff_jac_.dvn, f_y_off_diag_st_offset);
        }
        if (e.pos_diff_jac_.dv.has_block) {
            set_sp_mat(approx_->jac_[__x], e.pos_diff_jac_.dv, v_st, d.f_x_v_block, true);
            set_diag_zero(f_x_off_diag_, e.pos_diff_jac_.dv, f_x_off_diag_st_offset);
        }
        if (e.pos_diff_proj_jac_.dv.has_block) {
            set_sp_mat(dyn_proj_->proj_f_x_, e.pos_diff_proj_jac_.dv, v_st, d.proj_f_x_v_block, true);
            set_diag_zero(proj_f_x_off_diag_, e.pos_diff_proj_jac_.dv);
        }
        if (e.pos_diff_proj_jac_.da.has_block) {
            set_sp_mat(dyn_proj_->proj_f_u_, e.pos_diff_proj_jac_.da, a_st, d.proj_f_u_q_block);
            set_diag_zero(proj_f_u_q_diag_, e.pos_diff_proj_jac_.da, f_y_off_diag_st_offset);
        }
        setup_map(d.f_u_v_diag_, f_u_v_diag_.segment(e_st, e.v->dim()));
        if (e.v_int_type_ != euler::v_int_type::_implicit) {
            setup_map(d.f_x_v_off_diag_, f_x_off_diag_.segment(e_st - f_x_off_diag_st_offset, e.v->dim()));
        }
        if (e.v_int_type_ != euler::v_int_type::_explicit) {
            setup_map(d.f_y_v_off_diag_, f_y_off_diag_.segment(e_st - f_y_off_diag_st_offset, e.v->dim()));
        }
        if (dyn.has_timestep_) {
            setup_map(d.f_t_q, f_t.segment(e_st, e.q->tdim()));
            setup_map(d.f_t_v, f_t.segment(e_st + dyn.nv_, e.v->dim()));
            setup_map(d.proj_f_t_q, proj_f_t_.segment(e_st, e.q->tdim()));
        }
        e.setup_data(d);
        euler_data_.emplace_back(std::move(d));
        e_st += e.v->dim();
    }
}
void stacked_euler::value_impl(func_approx_data &data) const {
    auto &d = data.as<approx_data>();
    size_t i = 0;
    for (euler &e : dyn_) {
        e.value_impl(d.euler_data_[i++]);
    }
}
void stacked_euler::jacobian_impl(func_approx_data &data) const {
    auto &d = data.as<approx_data>();
    size_t i = 0;
    for (euler &e : dyn_) {
        e.jacobian_impl(d.euler_data_[i++]);
    }
    d.f_u_v_diag_.setConstant(-data[dt_](0));
}
void stacked_euler::compute_project_derivatives(func_approx_data &data) const {
    auto &d = data.as<approx_data>();
    size_t i = 0;
    d.proj_f_t_.array() = d.f_y_inv_main_diag_.array() * d.f_t.array();
    for (euler &e : dyn_) {
        e.compute_project_derivatives(d.euler_data_[i++]);
    }
    d.proj_f_u_v_diag_ = d.f_u_v_diag_;
    d.proj_f_x_off_diag_.setZero();
    size_t x_off_diag_dim = d.f_x_off_diag_.size();
    size_t y_off_diag_dim = d.f_y_off_diag_.size();
    if (x_off_diag_dim > 0)
        d.proj_f_x_off_diag_.head(x_off_diag_dim) += d.f_x_off_diag_;
    if (y_off_diag_dim > 0) {
        d.proj_f_x_off_diag_.tail(y_off_diag_dim) += d.f_y_off_diag_;
        d.f_y_inv_off_diag_.array() = -d.f_y_off_diag_.array();
        if (has_timestep_) {
            size_t y_off_diag_st = mid_st_ < max_dyn ? mid_st_ : imp_st_;
            d.proj_f_t_.segment(y_off_diag_st, y_off_diag_dim).noalias() += d.f_y_inv_off_diag_.cwiseProduct(d.f_t.tail(y_off_diag_dim));
            d.proj_f_u_q_diag_.noalias() = d.f_y_inv_off_diag_.cwiseProduct(d.f_u_v_diag_.tail(y_off_diag_dim));
        }
    }
    d.f_y_inv.times(d.approx_->v_, d.dyn_proj_->proj_f_res_);
}
void stacked_euler::apply_jac_y_inverse_transpose(func_approx_data &data, vector &v, vector &dst) const {
    auto &d = data.as<approx_data>();
    d.f_y_inv.T_times(v, dst);
}

} // namespace multibody

} // namespace moto