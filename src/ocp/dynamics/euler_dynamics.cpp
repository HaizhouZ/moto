#include <moto/ocp/dynamics/euler_dynamics.hpp>
#include <moto/ocp/problem.hpp>

namespace moto {
void explicit_euler_impl::finalize_impl() {
    add_arguments(first_ord_var_.pos_x_);
    add_arguments(first_ord_var_.pos_y_);
    add_arguments(first_ord_var_.vel_u_);
    add_arguments(sec_ord_var_.pos_x_);
    add_arguments(sec_ord_var_.vel_x_);
    add_arguments(sec_ord_var_.pos_y_);
    add_arguments(sec_ord_var_.vel_y_);
    add_arguments(sec_ord_var_.acc_u_);

    has_1st_ord_ = !first_ord_var_.pos_x_.empty();
    has_2nd_ord_ = !sec_ord_var_.pos_x_.empty();
    has_timestep_ = bool(timestep_var_);

    if (has_timestep_) {
        add_argument(timestep_var_);
    }

    generic_dynamics::finalize_impl();
}
void explicit_euler_impl::value_impl(func_approx_data &data) const {
    if (has_1st_ord_) {
        size_t idx = 0;
        for (size_t i = 0; i < first_ord_var_.pos_x_.size(); i++) {
            const auto &r = first_ord_var_.pos_x_[i];
            const auto &rn = first_ord_var_.pos_y_[i];
            const auto &v = first_ord_var_.vel_u_[i];
            if (has_timestep_) {
                data.v_.segment(idx, r->dim()) = data[rn] - data[r] - data[v] * dt_;
            } else {
                data.v_.segment(idx, r->dim()).noalias() = data[rn] - data[r] - data[v] * data[timestep_var_];
            }
            idx += r->dim();
        }
    }
    if (has_2nd_ord_) {
        size_t idx = 0;
        for (size_t i = 0; i < sec_ord_var_.pos_x_.size(); i++) {
            const auto &r = sec_ord_var_.pos_x_[i];
            const auto &rn = sec_ord_var_.pos_y_[i];
            const auto &v = sec_ord_var_.vel_x_[i];
            const auto &vn = sec_ord_var_.vel_y_[i];
            const auto &a = sec_ord_var_.acc_u_[i];
            size_t dim = r->dim();
            if (has_timestep_) {
                data.v_.segment(idx, dim) = data[rn] - data[r] - data[v] * dt_;
                data.v_.segment(idx + dim, dim) = data[vn] - data[v] - data[a] * dt_;
            } else {
                data.v_.segment(idx, dim).noalias() = data[rn] - data[r] - data[v] * data[timestep_var_];
                data.v_.segment(idx + dim, dim).noalias() = data[vn] - data[v] - data[a] * data[timestep_var_];
            }
            idx += r->dim();
        }
    }
}
void explicit_euler_impl::jacobian_impl(func_approx_data &data) const {
    auto &d = data.as<approx_data>();
    scalar_t dt = has_timestep_ ? d[timestep_var_](0) : dt_;
    d.f_u_.setConstant(-dt);
    d.f_x_off_diag_.setConstant(-dt);
    if (has_timestep_) {
        if (has_1st_ord_) {
            size_t idx = 0;
            for (size_t i = 0; i < first_ord_var_.pos_x_.size(); i++) {
                const auto &r = first_ord_var_.pos_x_[i];
                const auto &rn = first_ord_var_.pos_y_[i];
                const auto &v = first_ord_var_.vel_u_[i];
                d.f_dt_.segment(idx, r->dim()) = -d[v];
                idx += r->dim();
            }
        }
        if (has_2nd_ord_) {
            size_t idx = 0;
            for (size_t i = 0; i < sec_ord_var_.pos_x_.size(); i++) {
                const auto &r = sec_ord_var_.pos_x_[i];
                const auto &rn = sec_ord_var_.pos_y_[i];
                const auto &v = sec_ord_var_.vel_x_[i];
                const auto &vn = sec_ord_var_.vel_y_[i];
                const auto &a = sec_ord_var_.acc_u_[i];
                size_t dim = r->dim();
                d.f_dt_.segment(idx, dim) = -d[v];
                d.f_dt_.segment(idx + dim, dim) = -d[a];
                idx += r->dim();
            }
        }
    }
}
void explicit_euler_impl::compute_project_derivatives(func_approx_data &data) const {
    auto &d = data.as<approx_data>();
    scalar_t dt = has_timestep_ ? d[timestep_var_](0) : dt_;
    d.proj_f_x_off_diag_.setConstant(-dt);
    d.proj_f_u_.setConstant(-dt);
    if (has_timestep_) {
        d.proj_f_dt_ = d.f_dt_;
    }
}
explicit_euler_impl::approx_data::approx_data(generic_dynamics::approx_data &&rhs)
    : generic_dynamics::approx_data(std::move(rhs)),
      NULL_INIT_MAP(f_u_),
      NULL_INIT_MAP(f_x_off_diag_),
      NULL_INIT_MAP(f_dt_),
      NULL_INIT_MAP(proj_f_u_),
      NULL_INIT_MAP(proj_f_x_off_diag_),
      NULL_INIT_MAP(proj_f_dt_) {
    // create sparse pattern
    size_t f_st = problem()->get_expr_start(func_);
    auto &dyn = static_cast<const explicit_euler_impl &>(func_);
    array_type<size_t, primal_fields> arg_st{};
    size_t dim = func_.dim();
    assert(dim == func_.arg_dim(__x) && dim == func_.arg_dim(__y) &&
           "function dimension must match the dimensions of x and y");
    // setup jacobian
    approx_->jac_[__y].insert(f_st, arg_st[__y], dim, dim, sparsity::eye);
    {
        approx_->jac_[__x].insert(f_st, arg_st[__x], dim, dim, sparsity::diag).setConstant(-1.0);
        auto off_diag = approx_->jac_[__x].insert(f_st, arg_st[__x] + dyn.sec_ord_var_.dim_, dim, dyn.sec_ord_var_.dim_, sparsity::diag);
        new (&f_x_off_diag_) aligned_vector_map_t(off_diag.data(), off_diag.rows(), off_diag.cols());
    }
    {
        auto f_u = approx_->jac_[__u].insert(f_st + dyn.sec_ord_var_.dim_, arg_st[__u], func_.dim(), dim, sparsity::diag);
        new (&f_u_) aligned_vector_map_t(f_u.data(), f_u.rows(), f_u.cols());
    }
    if (dyn.has_timestep_) {
        auto f_dt = approx_->jac_[__u].insert(f_st, problem()->get_expr_start(dyn.timestep_var_), func_.dim(), 1, sparsity::dense);
        new (&f_dt_) aligned_vector_map_t(f_dt.data(), f_dt.rows(), f_dt.cols());
    }

    // set projected dynamics derivatives
    {
        auto m_diag = dyn_proj_->proj_f_x_.insert(f_st, arg_st[__x], dim, dim, sparsity::diag).setConstant(-1.0);
        auto off_diag = dyn_proj_->proj_f_x_.insert(f_st, arg_st[__x] + dyn.sec_ord_var_.dim_, dim, dyn.sec_ord_var_.dim_, sparsity::diag);
        new (&proj_f_x_off_diag_) aligned_vector_map_t(off_diag.data(), off_diag.rows(), off_diag.cols());
    }
    {
        auto f_u = dyn_proj_->proj_f_u_.insert(f_st + dyn.sec_ord_var_.dim_, arg_st[__u], func_.dim(), dim, sparsity::diag);
        new (&proj_f_u_) aligned_vector_map_t(f_u.data(), f_u.rows(), f_u.cols());
    }
    if (dyn.has_timestep_) {
        auto f_dt = dyn_proj_->proj_f_u_.insert(f_st, problem()->get_expr_start(dyn.timestep_var_), func_.dim(), 1, sparsity::dense);
        new (&proj_f_dt_) aligned_vector_map_t(f_dt.data(), f_dt.rows(), f_dt.cols());
    }
}
} // namespace moto