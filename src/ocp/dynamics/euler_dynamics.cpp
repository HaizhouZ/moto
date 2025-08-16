#include <moto/ocp/dynamics/euler_dynamics.hpp>
#include <moto/ocp/problem.hpp>

namespace moto {
void explicit_euler::impl::finalize_impl() {
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

    dim_ = first_ord_var_.dim_ + sec_ord_var_.dim_ * 2;

    if (has_timestep_) {
        add_argument(timestep_var_);
    }

    generic_dynamics::finalize_impl();
}
void explicit_euler::impl::value_impl(func_approx_data &data) const {
    if (has_1st_ord_) {
        size_t idx = 0;
        for (size_t i = 0; i < first_ord_var_.pos_x_.size(); i++) {
            const auto &r = first_ord_var_.pos_x_[i];
            const auto &rn = first_ord_var_.pos_y_[i];
            const auto &v = first_ord_var_.vel_u_[i];
            if (!has_timestep_) {
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
            if (!has_timestep_) {
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
void explicit_euler::impl::jacobian_impl(func_approx_data &data) const {
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
void explicit_euler::impl::compute_project_derivatives(func_approx_data &data) const {
    auto &d = data.as<approx_data>();
    scalar_t dt = has_timestep_ ? d[timestep_var_](0) : dt_;
    d.proj_f_x_off_diag_.setConstant(-dt);
    d.proj_f_u_.setConstant(-dt);
    if (has_timestep_) {
        d.proj_f_dt_ = d.f_dt_;
    }
}
explicit_euler::impl::approx_data::approx_data(generic_dynamics::approx_data &&rhs)
    : generic_dynamics::approx_data(std::move(rhs)),
      NULL_INIT_VECMAP(f_u_),
      NULL_INIT_VECMAP(f_x_off_diag_),
      NULL_INIT_VECMAP(f_dt_),
      NULL_INIT_VECMAP(proj_f_u_),
      NULL_INIT_VECMAP(proj_f_x_off_diag_),
      NULL_INIT_VECMAP(proj_f_dt_) {
    // create sparse pattern
    size_t f_st = problem()->get_expr_start(func_);
    auto &dyn = static_cast<const explicit_euler::impl &>(func_);
    array_type<size_t, primal_fields> arg_st{};
    const var &first_x = dyn.first_ord_var_.dim_ ? dyn.first_ord_var_.pos_x_[0] : dyn.sec_ord_var_.pos_x_[0];
    const var &first_y = dyn.first_ord_var_.dim_ ? dyn.first_ord_var_.pos_y_[0] : dyn.sec_ord_var_.pos_y_[0];
    const var &first_u = dyn.first_ord_var_.dim_ ? dyn.first_ord_var_.vel_u_[0] : dyn.sec_ord_var_.acc_u_[0];
    arg_st[__x] = problem()->get_expr_start(first_x);
    arg_st[__y] = problem()->get_expr_start(first_y);
    arg_st[__u] = problem()->get_expr_start(first_u);
    size_t dim = func_.dim();
    assert(dim == func_.arg_dim(__x) && dim == func_.arg_dim(__y) &&
           "function dimension must match the dimensions of x and y");
    size_t dim_1st = dyn.first_ord_var_.dim_;
    size_t dim_2nd = dyn.sec_ord_var_.dim_;
    // setup jacobian
    approx_->jac_[__y].insert(f_st, arg_st[__y], dim, dim, sparsity::eye);
    {
        approx_->jac_[__x].insert(f_st, arg_st[__x], dim, dim, sparsity::diag).setConstant(-1.0);
        auto off_diag = approx_->jac_[__x].insert(f_st, arg_st[__x] + dim_2nd, dim_2nd, dim_2nd, sparsity::diag);
        setup_map(f_x_off_diag_, off_diag);
        dyn_proj_->proj_f_x_.insert(f_st, arg_st[__x], dim, dim, sparsity::diag).setConstant(-1.0);
        auto proj_off_diag = dyn_proj_->proj_f_x_.insert(f_st, arg_st[__x] + dim_2nd, f_x_off_diag_.rows(), f_x_off_diag_.cols(), sparsity::diag);
        setup_map(proj_f_x_off_diag_, proj_off_diag);
    }
    {
        auto f_u = approx_->jac_[__u].insert(f_st + dim_2nd, arg_st[__u], dim_1st + dim_2nd, dim_1st + dim_2nd, sparsity::diag);
        setup_map(f_u_, f_u);
        auto proj_f_u = dyn_proj_->proj_f_u_.insert(f_st + dim_2nd, arg_st[__u], f_u_.rows(), f_u_.cols(), sparsity::diag);
        setup_map(proj_f_u_, proj_f_u);
    }
    if (dyn.has_timestep_) {
        auto f_dt = approx_->jac_[__u].insert(f_st, problem()->get_expr_start(dyn.timestep_var_), func_.dim(), 1, sparsity::dense);
        setup_map(f_dt_, f_dt);
        auto proj_f_dt = dyn_proj_->proj_f_u_.insert(f_st, problem()->get_expr_start(dyn.timestep_var_), func_.dim(), 1, sparsity::dense);
        setup_map(proj_f_dt_, proj_f_dt);
    }
}
} // namespace moto