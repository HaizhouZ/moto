#include <moto/core/external_function.hpp>
#include <moto/multibody/euler.hpp>
#include <moto/ocp/dynamics.hpp>
#include <moto/ocp/problem.hpp>
#include <moto/utils/codegen.hpp>
#include <Eigen/Dense>

namespace moto {
namespace multibody {
void euler::jac_block::setup(const cs::SX &mat, size_t c_offset) {
    dense = mat;
    assert(dense.is_square() && "the jacobian must be square");
    size_t nnz_total = dense.nnz();
    diag = cs::SX::sparsify(cs::SX::diag(dense));
    dense = cs::SX::sparsify(dense - diag);
    assert(dense.nnz() + diag.nnz() == nnz_total && "sparsify error");
    empty = dense.nnz() == 0 || diag.nnz() == 0; /// @todo append dense to diag
    has_block = dense.nnz() > 0;
    if (has_block) {
        size_t dense_nnz = dense.nnz();
        std::vector<casadi_int> rows(dense_nnz), cols(dense_nnz);
        dense.sparsity().get_triplet(rows, cols);
        auto [minr, maxr] = std::minmax_element(rows.begin(), rows.end());
        auto [minc, maxc] = std::minmax_element(cols.begin(), cols.end());
        dense_block.r_st = *minr;
        dense_block.nrow = *maxr - *minr + 1;
        dense_block.c_st = c_offset + *minc;
        dense_block.ncol = *maxc - *minc + 1;
        dense += diag(cs::Slice(dense_block.r_st, dense_block.nrow));
        diag(cs::Slice(dense_block.r_st, dense_block.nrow)) = 0.;
    }
}
euler::euler(const std::string &name,
             const var &q, const var &v, const var &a, const var &dt, cs::SX pos_step,
             cs::SX pos_diff, cs::SX dpos_diff, cs::SX pos_int, cs::SX dpos_int)
    : base(name,
           {q, v, q->next(), v->next(), a, dt},
           cs::SX::vertcat({pos_diff, v->next() - (v + a * dt)}),
           approx_order::first, __dyn),
      q_(q), v_(v), a_(a), dt_(dt),
      pos_integrate_(pos_int), pos_diff_(pos_diff), pos_step_(pos_step),
      dpos_diff_{dpos_diff, dpos_int}, dpos_int_{dpos_int, dpos_int} {
    q_->tdim() = pos_diff_.size1();
    assert(pos_integrate_.size1() == q_->dim() && "the position step must have the same dimension as q");
    assert(v_->dim() == q_->tdim() && "the velocity dimension must match the tangent space dimension of q");
}
void euler::load_external_impl(const std::string &path) {
    generic_func::load_external_impl(path);
    auto f = ext_func(gen_.task_->extra_task->func_name);
    q_->integrator() = [f_ = std::move(f)](vector_ref x, vector_ref dx, vector_ref out, scalar_t alpha) {
        std::vector<vector_ref> args = {x, dx, mapped_vector(&alpha, 1)};
        f_.invoke(args, out);
    };
}
void euler::finalize_impl() {
    pos_diff_jac_.dqn.setup(dpos_diff_[0]);
    pos_diff_jac_.dq.setup(dpos_diff_[1] * dpos_int_[0]);
    auto dpos_diff_dstep = dpos_diff_[1] * dpos_int_[1];
    pos_diff_jac_.dvn.setup(dpos_diff_dstep * cs::SX::jacobian(pos_step_, v_->next()), q_->tdim());
    pos_diff_jac_.dv.setup(dpos_diff_dstep * cs::SX::jacobian(pos_step_, v_), q_->tdim());
    // setup the projection jacobian
    // the inverse of F_y is the same as F_y, maybe buggy if the dqn and dq are not consistent
    if (pos_diff_jac_.dqn.has_block)
        pos_diff_proj_jac_.dq.dense_block = pos_diff_jac_.dqn.dense_block;
    else if (pos_diff_jac_.dq.has_block)
        pos_diff_proj_jac_.dq.dense_block = pos_diff_jac_.dq.dense_block;
    if (pos_diff_jac_.dvn.has_block)
        pos_diff_proj_jac_.dv.dense_block = pos_diff_jac_.dvn.dense_block;
    else if (pos_diff_jac_.dv.has_block)
        pos_diff_proj_jac_.dv.dense_block = pos_diff_jac_.dv.dense_block;
    // setup proj input jacobian
    if (pos_diff_jac_.dvn.has_block)
        pos_diff_proj_jac_.da.dense_block = pos_diff_jac_.dvn.dense_block;
    // these sparsity are enough to setup the spmat
    if (!pos_diff_jac_.dvn.empty) {
        if (pos_diff_jac_.dv.empty) {
            v_int_type_ = v_int_type::__implicit;
        } else {
            v_int_type_ = v_int_type::__mid_point;
        }
    }
    // generate the casadi integrator
    utils::cs_codegen::task integrator_task;
    integrator_task.func_name = name_ + "_int";
    integrator_task.sx_inputs = {q_, v_, dt_}; // v_ here just represent the step in the integrator
    integrator_task.sx_output = pos_integrate_;
    integrator_task.keep_generated_src = true;
    // setup the sparse jacs
    auto &jacs = gen_.task_->jac_outputs;
    jacs.emplace_back(pos_diff_jac_.dqn.dense);
    jacs.emplace_back(pos_diff_jac_.dq.dense);
    jacs.emplace_back(pos_diff_jac_.dvn.dense);
    jacs.emplace_back(pos_diff_jac_.dv.dense);
    jacs.emplace_back(pos_diff_jac_.dvn.diag);
    jacs.emplace_back(pos_diff_jac_.dv.diag);
    if (dt_->field() == __u) {
        jacs.emplace_back(cs::SX::jacobian(pos_integrate_, dt_));
    }
    gen_.task_->extra_task.reset(new utils::cs_codegen::task(std::move(integrator_task)));
    base::finalize_impl();
};

void euler::setup_data(euler_data &data) const {
    size_t i = 0;
    new (&data.jac_[i++]) matrix_ref(data.f_y_q_block);
    new (&data.jac_[i++]) matrix_ref(data.f_x_q_block);
    new (&data.jac_[i++]) matrix_ref(data.f_y_v_block);
    new (&data.jac_[i++]) matrix_ref(data.f_x_v_block);
    new (&data.jac_[i++]) matrix_ref(data.f_y_v_off_diag_);
    new (&data.jac_[i++]) matrix_ref(data.f_x_v_off_diag_);
    if (dt_->field() == __u) {
        new (&data.jac_[i++]) matrix_ref(data.f_t_q);
    }
}

void euler::jacobian_impl(func_approx_data &data) const {
    base::jacobian_impl(data);
    auto &d = data.as<euler_data>();
    d.f_t_v.array() = -data[a_];
}

void euler::compute_project_derivatives(euler_data &data) const {
    if (pos_diff_jac_.dvn.has_block) {
        data.f_y_inv_v_block.array() = -data.f_y_v_block.array();
    }
    if (pos_diff_jac_.dqn.has_block) {
        data.f_y_inv_q_block.noalias() = data.f_y_q_block.inverse();
    }
    if (pos_diff_jac_.dqn.has_block) {
        if (pos_diff_jac_.dq.has_block) {
            data.proj_f_x_q_block.noalias() = data.f_y_inv_q_block * data.f_x_q_block;
        } else {
            data.proj_f_x_q_block.array() = -data.f_y_inv_q_block.array();
        }
    }
    if (pos_diff_jac_.dvn.has_block) {
        data.proj_f_x_v_block.array() += data.f_y_v_block.array(); // f_y diag is -1, cancel with -1 in proj
    }
    if (pos_diff_jac_.dv.has_block) {
        if (pos_diff_jac_.dqn.has_block) {
            data.proj_f_x_v_block.noalias() += data.f_y_inv_q_block * data.f_x_v_block;
        } else
            data.proj_f_x_v_block += data.f_x_v_block;
    }
    if (pos_diff_proj_jac_.da.has_block) {
        size_t st = pos_diff_proj_jac_.da.dense_block.c_st;
        size_t n = pos_diff_proj_jac_.da.dense_block.ncol;
        data.proj_f_u_q_block.noalias() = data.f_y_inv_v_block * data.f_u_v_diag_.segment(st, n).asDiagonal();
    }
    if (dt_->field() == __u) {
        if (pos_diff_jac_.dqn.has_block) {
            data.proj_f_t_q.noalias() += data.f_y_inv_q_block * data.f_t_q;
        }
        if (pos_diff_jac_.dvn.has_block) {
            data.proj_f_t_q.noalias() += data.f_y_inv_v_block * data.f_t_v;
        }
    }
}

void stacked_euler::finalize_impl() {
    // make sure the arg order is [pos, vel] !
    std::sort(dyn_.begin(), dyn_.end(), [](const euler &a, const euler &b) {
        return a.v_int_type_ < b.v_int_type_;
    });
    dt_ = ((euler &)dyn_[0]).dt_;
    has_timestep_ = dt_->field() == __u;
    for (const euler &e : dyn_) {
        if (dt_ != e.dt_) {
            throw std::runtime_error("all euler dynamics must have the same time step variable");
        }
        if (e.v_int_type_ == euler::v_int_type::__implicit) {
            imp_st_ = std::min(imp_st_, nv_);
            dim_imp_ += e.v_->dim();
        } else if (e.v_int_type_ == euler::v_int_type::__mid_point) {
            exp_st_ = std::min(exp_st_, nv_);
            dim_mid_ += e.v_->dim();
        } else if (e.v_int_type_ == euler::v_int_type::__explicit) {
            mid_st = std::min(mid_st, nv_);
            dim_exp_ += e.v_->dim();
        }
        add_argument(e.q_);
        add_argument(e.a_);
        nq_ += e.q_->dim();
        nv_ += e.v_->dim();
    }
    for (const euler &e : dyn_) {
        add_argument(e.v_);
    }
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
    vector_ref off_diag_y_inv = f_y_inv.insert<sparsity::diag>(f_st, y_st + dyn.nv_, dyn.nv_);
    setup_map(f_y_inv_off_diag_, off_diag_y_inv);
    vector_ref off_diag_x = approx_->jac_[__x].insert<sparsity::diag>(f_st + dyn.exp_st_, x_st + dyn.nv_ + dyn.exp_st_, dyn.dim_exp_ + dyn.dim_mid_);
    setup_map(f_x_off_diag_, off_diag_x);
    vector_ref proj_off_diag_x = dyn_proj_->proj_f_x_.insert<sparsity::diag>(f_st + dyn.exp_st_, x_st + dyn.nv_, dyn.nv_);
    setup_map(proj_f_x_off_diag_, proj_off_diag_x);
    vector_ref off_diag_y = approx_->jac_[__y].insert<sparsity::diag>(f_st + dyn.exp_st_, x_st + dyn.nv_ + dyn.mid_st, dyn.dim_imp_ + dyn.dim_mid_);
    setup_map(f_y_off_diag_, off_diag_y);
    vector_ref u_v_diag = approx_->jac_[__u].insert<sparsity::diag>(f_st + dyn.nv_, u_st, dyn.nv_);
    setup_map(f_u_v_diag_, u_v_diag);
    // setup the projection of input jacobian diag
    vector_ref proj_u_diag_q = dyn_proj_->proj_f_u_.insert<sparsity::diag>(f_st, u_st, dyn.nv_);
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
    for (euler &e : dyn.dyn_) {
        size_t q_st = prob.get_expr_start_tangent(e.q_);
        size_t v_st = prob.get_expr_start_tangent(e.v_);
        size_t qn_st = prob.get_expr_start_tangent(e.q_->next());
        size_t vn_st = prob.get_expr_start_tangent(e.v_->next());
        size_t a_st = prob.get_expr_start_tangent(e.a_);
        euler_data d(*this->primal_, *this->merit_data_, this->shared_, e);
        auto set_sp_mat = [&, this](sparse_mat &dst, euler::jac_block &jb, size_t c_st, auto &_map, bool is_vel = false) {
            auto m = dst.insert(f_st + e_st + jb.dense_block.r_st,
                                c_st + jb.dense_block.c_st + (is_vel ? dyn.nv_ : 0),
                                jb.dense_block.nrow,
                                jb.dense_block.ncol, sparsity::dense);
            setup_map(_map, m);
        };
        auto set_diag_zero = [&, this](auto &diag_mat, euler::jac_block &jb) {
            diag_mat.segment(e_st + jb.dense_block.r_st, jb.dense_block.nrow).setZero();
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
            set_sp_mat(approx_->jac_[__y], e.pos_diff_jac_.dq, q_st, d.f_x_q_block);
            set_diag_zero(f_x_main_diag, e.pos_diff_jac_.dq);
        }
        if (e.pos_diff_jac_.dvn.has_block) {
            set_sp_mat(approx_->jac_[__y], e.pos_diff_jac_.dvn, vn_st, d.f_y_v_block, true);
            set_sp_mat(f_y_inv, e.pos_diff_jac_.dvn, v_st, d.f_y_inv_v_block, true);
            set_diag_zero(f_y_off_diag_, e.pos_diff_jac_.dvn);
        }
        if (e.pos_diff_jac_.dv.has_block) {
            set_sp_mat(approx_->jac_[__y], e.pos_diff_jac_.dv, v_st, d.f_x_v_block, true);
            set_diag_zero(f_x_off_diag_, e.pos_diff_jac_.dv);
        }
        if (e.pos_diff_proj_jac_.dv.has_block) {
            set_sp_mat(dyn_proj_->proj_f_x_, e.pos_diff_proj_jac_.dv, v_st, d.proj_f_x_v_block, true);
            set_diag_zero(proj_f_x_off_diag_, e.pos_diff_proj_jac_.dv);
        }
        if (e.pos_diff_proj_jac_.da.has_block) {
            set_sp_mat(dyn_proj_->proj_f_u_, e.pos_diff_proj_jac_.da, a_st, d.proj_f_u_q_block);
            set_diag_zero(proj_f_u_q_diag_, e.pos_diff_proj_jac_.da);
        }
        setup_map(d.f_u_v_diag_, f_u_v_diag_.segment(e_st, e.v_->dim()));
        if (e.v_int_type_ != euler::v_int_type::__implicit)
            setup_map(d.f_x_v_off_diag_, f_x_off_diag_.segment(e_st, e.v_->dim()));
        if (e.v_int_type_ != euler::v_int_type::__explicit) {
            setup_map(d.f_y_v_off_diag_, f_y_off_diag_.segment(e_st, e.v_->dim()));
        }
        if (dyn.has_timestep_) {
            setup_map(d.f_t_q, f_t.segment(e_st, e.dim()));
            setup_map(d.f_t_v, f_t.segment(e_st + dyn.nv_, e.dim()));
            setup_map(d.proj_f_t_q, proj_f_t_.segment(e_st, e.dim()));
        }
        euler_data_.emplace_back(std::move(d));
        e_st += dyn.nv_;
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
    if (dim_exp_ + dim_imp_ > 0)
        d.proj_f_x_off_diag_.head(dim_exp_ + dim_imp_) += d.f_x_off_diag_;
    if (dim_mid_ + dim_imp_ > 0)
        d.proj_f_x_off_diag_.tail(dim_mid_ + dim_imp_) += d.f_y_off_diag_;
    d.f_y_inv_off_diag_.array() = -d.f_y_off_diag_.array();
    d.proj_f_u_q_diag_.array() = d.f_y_inv_off_diag_.array() * d.f_u_v_diag_.array();
}
void stacked_euler::apply_jac_y_inverse_transpose(func_approx_data &data, vector& v, vector& dst) const {
    auto &d = data.as<approx_data>();
    d.f_y_inv.T_times(v, dst);
}

} // namespace multibody
} // namespace moto