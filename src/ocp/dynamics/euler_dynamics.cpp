#include <Eigen/Geometry>
#include <moto/ocp/dynamics/euler_dynamics.hpp>
#include <moto/ocp/problem.hpp>
namespace moto {

struct dynamics_with_dt {
    virtual size_t get_dim() const = 0;
    virtual void value(size_t st, func_approx_data &data, scalar_t dt) const = 0; ///< compute value at time step
    virtual void f_dt(size_t st, func_approx_data &data, vector_ref f_dt_) const = 0;
    virtual void add_proj_f_dt(size_t st, func_approx_data &data, scalar_t dt) const {};
};

class euler_impl : public generic_dynamics {
  public:
    struct approx_data : public generic_dynamics::approx_data {
        aligned_vector_map_t f_y_lin_off_diag_, f_x_lin_off_diag_, proj_f_x_lin_off_diag_;
        std::vector<aligned_map_t> f_ang_y_off_diag_, f_ang_x_off_diag_, proj_f_ang_x_off_diag_, proj_f_ang_u_off_diag_;
        aligned_vector_map_t f_u_, proj_f_u_;
        aligned_vector_map_t f_dt_, proj_f_dt_;
        approx_data(generic_dynamics::approx_data &&rhs);
    };

  private:
    bool has_timestep_ = false; ///< true if timestep variable is present
    scalar_t dt_ = 0.0;         ///< time step
    var timestep_var_;          ///< time variable
    scalar_t get_dt(func_approx_data &data) const { return has_timestep_ ? data[timestep_var_](0) : dt_; }

    bool has_1st_ord_lin_ = false; ///< true if first order dynamics are present
    bool has_2nt_ord_lin_ = false; ///< true if second order dynamics are present
    struct sec_ord_ang : public dynamics_with_dt {
        size_t dim_q_ = 0;           ///< dimension of the state
        size_t dim_w_ = 0;           ///< dimension of the local angular velocity
        var quat_x_;                 ///< quaternion cur
        var quat_y_;                 ///< quaternion next
        var w_x_;                    ///< angular velocity cur
        var w_y_;                    ///< angular velocity next
        var a_u_;                    ///< angular acceleration input
        bool semi_implicit_ = false; ///< true if semi-implicit euler
        size_t local_idx = 0;        ///< local index in angular dynamics
        sec_ord_ang(const var &q, const var &qn, const var &w, const var &wn, const var &a)
            : quat_x_(q), quat_y_(qn), w_x_(w), w_y_(wn), a_u_(a) {
            dim_q_ = q->dim();
            dim_w_ = w->dim();
        }
        size_t get_dim() const override { return dim_q_ + dim_w_; } ///< get dimension of the state
        void value(size_t st, func_approx_data &data, scalar_t dt) const override {
            auto &d = data.as<approx_data>();
            if (semi_implicit_) {
                set_transform(data, d.f_ang_y_off_diag_[local_idx]);
                auto w_to_int = data[w_y_];
                data.v_.segment(st, dim_q_) = data[quat_y_] - data[quat_x_] - d.f_ang_y_off_diag_[local_idx] * w_to_int * dt;
            } else {
                auto w_to_int = data[w_x_];
                set_transform(data, d.f_ang_x_off_diag_[local_idx]);
                data.v_.segment(st, dim_q_) = data[quat_y_] - data[quat_x_] + d.f_ang_x_off_diag_[local_idx] * w_to_int * dt;
            }
            data.v_.segment(st + dim_q_, dim_w_) = data[w_y_] - data[w_x_] - data[a_u_] * dt;
        }
        void f_dt(size_t st, func_approx_data &data, vector_ref f_dt_) const override {
            auto &d = data.as<approx_data>();
            if (semi_implicit_) {
                f_dt_.segment(st, dim_q_) = -d.f_ang_y_off_diag_[local_idx] * data[w_y_];
            } else {
                f_dt_.segment(st, dim_q_) = d.f_ang_x_off_diag_[local_idx] * data[w_x_];
            }
            f_dt_.segment(st + dim_q_, dim_w_) = -data[a_u_];
        } ///< compute f_dt for this dynamics
        void add_proj_f_dt(size_t st, func_approx_data &data, scalar_t dt) const override {
            auto &d = data.as<approx_data>();
            d.f_dt_.segment(st, dim_q_).noalias() -= d.f_ang_y_off_diag_[local_idx] * dt * d.proj_f_dt_.segment(st + dim_q_, dim_w_);
        }
        void set_transform(func_approx_data &data, approx_data::aligned_map_t &H_tr) const {
            Eigen::Quaternion<scalar_t> q(data[quat_y_].data());
            // H_local << -q.x(), q.w(), q.z(), -q.y(),
            //     -q.y(), -q.z(), q.w(), q.x(),
            //     -q.z(), q.y(), -q.x(), q.w();
            matrix_rm::AlignedMapType H_local(H_tr.data(), 3, 4);
            H_local << -q.x(), q.w(), q.z(), -q.y(),
                -q.y(), -q.z(), q.w(), q.x(),
                -q.z(), q.y(), -q.x(), q.w();
            if (semi_implicit_) {
                H_local *= 0.5; ///< semi-implicit euler has a factor of 0.5
            } else {
                H_local *= -0.5; ///< explicit euler has a factor of -0.5
            }
        }
    };
    std::list<sec_ord_ang> sec_ord_ang_;    ///< second order angular variables
    std::list<sec_ord_ang> sec_ord_ang_si_; ///< second order angular variables (semi-implicit)
    struct sec_ord_lin : public dynamics_with_dt {
        size_t dim_ = 0;             ///< dimension of the state
        var pos_x_;                  ///< position variables
        var pos_y_;                  ///< position variables
        var vel_x_;                  ///< velocity variables
        var vel_y_;                  ///< velocity variables
        var acc_u_;                  ///< acceleration variables
        bool semi_implicit_ = false; ///< true if semi-implicit euler
        sec_ord_lin(const var &r, const var &rn, const var &v, const var &vn, const var &a, bool semi_implicit)
            : pos_x_(r), pos_y_(rn), vel_x_(v), vel_y_(vn), acc_u_(a) {
            dim_ = r->dim();
        }
        size_t get_dim() const override { return dim_ * 2; } ///< get dimension of the state
        void value(size_t st, func_approx_data &data, scalar_t dt) const override {
            auto v_to_int = semi_implicit_ ? data[vel_y_] : data[vel_x_];
            data.v_.segment(st, dim_).noalias() = data[pos_y_] - data[pos_x_] - v_to_int * dt;
            data.v_.segment(st + dim_, dim_).noalias() = data[vel_y_] - data[vel_x_] - data[acc_u_] * dt;
        }
        void f_dt(size_t st, func_approx_data &data, vector_ref f_dt_) const override {
            f_dt_.segment(st, dim_) = -data[vel_x_];
            f_dt_.segment(st + dim_, dim_) = -data[acc_u_];
        } ///< compute f_dt for this dynamics
        void add_proj_f_dt(size_t st, func_approx_data &data, scalar_t dt) const override {
            auto &d = data.as<approx_data>();
            d.proj_f_dt_.segment(st, dim_).noalias() -= dt * d.proj_f_dt_.segment(st + dim_, dim_); ///< semi-implicit euler
        }
    };
    std::list<sec_ord_lin> sec_ord_lin_si_; ///< second order variables semi-implicit
    std::list<sec_ord_lin> sec_ord_lin_;    ///< second order variables

    struct fst_ord_lin : public dynamics_with_dt {
        size_t dim_ = 0; ///< dimension of the state
        var pos_x_;      ///< position variables in state field
        var pos_y_;      ///< position variables in state field
        var vel_u_;      ///< velocity variables in control field
        fst_ord_lin(const var &r, const var &rn, const var &v) : pos_x_(r), pos_y_(rn), vel_u_(v) {
            dim_ = r->dim();
        }
        size_t get_dim() const override { return dim_; } ///< get dimension of the state
        void value(size_t st, func_approx_data &data, scalar_t dt) const override {
            data.v_.segment(st, dim_) = data[pos_y_] - data[pos_x_] - data[vel_u_] * dt;
        }
        void f_dt(size_t st, func_approx_data &data, vector_ref f_dt_) const override {
            f_dt_.segment(st, dim_) = -data[vel_u_];
        } ///< compute f_dt for this dynamics
    };
    std::list<fst_ord_lin> fst_ord_lin_; ///< first order variables

    std::list<std::reference_wrapper<dynamics_with_dt>> all_dyn_; ///< all dynamics variables for iteration

    size_t n_state_sec_ord_ang_ = 0;    ///< dimension of second order angular dynamics
    size_t n_state_sec_ord_si_ang_ = 0; ///< dimension of second order angular dynamics
    size_t n_state_sec_ord_lin_ = 0;    ///< dimension of second order linear dynamics
    size_t n_state_sec_ord_si_lin_ = 0; ///< dimension of second order linear dynamics
    size_t n_state_fst_ord_lin_ = 0;    ///< dimension of first order linear

    virtual void finalize_impl() override;
    friend struct euler; ///< allow euler to access private members

  public:
    func_approx_data_ptr_t create_approx_data(sym_data &primal,
                                              merit_data &raw,
                                              shared_data &shared) const override {
        return func_approx_data_ptr_t(make_approx<euler_impl>(primal, raw, shared));
    }
    void value_impl(func_approx_data &data) const override;
    void jacobian_impl(func_approx_data &data) const override;

    euler_impl(const std::string &name)
        : generic_dynamics(name, approx_order::first, dim_tbd, __dyn) {}

    void compute_project_derivatives(func_approx_data &data) const override;
};

#define IMPL static_cast<euler_impl *>((*this).operator->())

packed_retval<var, 5> euler::create_2nd_ord_vars(const std::string &name, size_t dim, bool semi_implicit) {
    auto [r, rn] = sym::states(name, dim);
    auto [v, vn] = sym::states(name + "_vel", dim);
    auto a = sym::inputs(name + "_acc", dim);
    if (semi_implicit) {
        IMPL->sec_ord_lin_.emplace_back(r, rn, v, vn, a, semi_implicit);
        IMPL->n_state_sec_ord_lin_ += dim;
    } else {
        IMPL->sec_ord_lin_si_.emplace_back(r, rn, v, vn, a, semi_implicit);
        IMPL->n_state_sec_ord_si_lin_ += dim;
    }

    return {std::move(r), std::move(rn), std::move(v), std::move(vn), std::move(a)};
}

packed_retval<var, 3> euler::create_1st_ord_vars(const std::string &name, size_t dim) {
    auto [r, rn] = sym::states(name, dim);
    auto v = sym::inputs(name + "_vel", dim);
    IMPL->fst_ord_lin_.emplace_back(r, rn, v);
    IMPL->n_state_fst_ord_lin_ += dim;
    return {std::move(r), std::move(rn), std::move(v)};
}

void euler_impl::finalize_impl() {
    // make sure the order is right!
    // add positional variables
    for (auto *ds : {&sec_ord_lin_si_, &sec_ord_lin_})
        for (auto &d : *ds) {
            add_arguments({d.pos_x_, d.pos_y_});
            all_dyn_.emplace_back(d);
        }

    for (auto *ds : {&sec_ord_ang_si_, &sec_ord_ang_}) {
        size_t idx = 0;
        for (auto &d : *ds) {
            d.local_idx = idx++;
            add_arguments({d.quat_x_, d.quat_y_});
            all_dyn_.emplace_back(d);
        }
    }

    // add velocity variables
    for (auto *ds : {&sec_ord_lin_si_, &sec_ord_lin_})
        for (auto &d : *ds)
            add_arguments({d.vel_x_, d.vel_y_});

    for (auto *ds : {&sec_ord_ang_si_, &sec_ord_ang_})
        for (auto &d : *ds)
            add_arguments({d.w_x_, d.w_y_});

    // add acceleration variables
    for (auto *ds : {&sec_ord_lin_si_, &sec_ord_lin_})
        for (auto &d : *ds)
            add_arguments({d.acc_u_});

    for (auto *ds : {&sec_ord_ang_si_, &sec_ord_ang_})
        for (auto &d : *ds)
            add_arguments({d.a_u_});

    // fisrt order variables
    for (auto &d : fst_ord_lin_) {
        add_arguments({d.pos_x_, d.pos_y_, d.vel_u_});
        all_dyn_.emplace_back(d);
    }

    dim_ = 0;
    for (dynamics_with_dt &d : all_dyn_) {
        dim_ += d.get_dim();
    }
    if (has_timestep_) {
        add_argument(timestep_var_);
    }
    if (!has_timestep_ && dt_ <= 0.0)
        throw std::runtime_error("Time step must be set before finalizing dynamics");

    generic_dynamics::finalize_impl();

    if (!(dim_ == arg_dim(__x) && dim_ == arg_dim(__y))) {
        throw std::runtime_error(fmt::format("Euler dynamics {} has dimension mismatch: expected {}, got x: {} y : {}",
                                             name_, dim_, arg_dim(__x), arg_dim(__y)));
    }
}
void euler_impl::value_impl(func_approx_data &data) const {
    size_t idx = 0;
    scalar_t dt = get_dt(data);
    for (const dynamics_with_dt &d : all_dyn_) {
        d.value(idx, data, dt); ///< compute value at time step
        idx += d.get_dim();
    }
}
void euler_impl::jacobian_impl(func_approx_data &data) const {
    auto &d = data.as<approx_data>();
    scalar_t dt = get_dt(data);
    d.f_u_.setConstant(-dt);
    d.f_y_lin_off_diag_.setConstant(dt);
    d.f_x_lin_off_diag_.setConstant(-dt);
    if (has_timestep_) {
        size_t idx = 0;
        for (const dynamics_with_dt &f : all_dyn_) {
            f.f_dt(idx, data, d.f_dt_); ///< compute f_dt for this dynamics
            idx += f.get_dim();
        }
    }
}
void euler_impl::compute_project_derivatives(func_approx_data &data) const {
    auto &d = data.as<approx_data>();
    scalar_t dt = get_dt(data);
    d.proj_f_x_lin_off_diag_.setConstant(-dt);
    d.proj_f_u_.setConstant(-dt);
    // d.proj_f_res_ = d.v_;
    /// @todo ang and semi-impliciit res
    for (auto *fs : {&sec_ord_ang_si_, &sec_ord_ang_}) {
        for (auto &f : *fs) {
            if (f.semi_implicit_) {
                d.proj_f_ang_x_off_diag_[f.local_idx] = -d.f_ang_y_off_diag_[f.local_idx];
                d.proj_f_ang_u_off_diag_[f.local_idx] = d.proj_f_ang_x_off_diag_[f.local_idx];
            } else {
                d.proj_f_ang_x_off_diag_[f.local_idx] = d.f_ang_x_off_diag_[f.local_idx];
            }
        }
    }
    if (has_timestep_) {
        d.proj_f_dt_ = d.f_dt_;
        size_t idx = 0;
        // correct semi-implicit dynamics
        for (auto &f : sec_ord_lin_si_) {
            f.add_proj_f_dt(idx, data, dt); ///< add projection of f_dt
            idx += f.get_dim();
        }
        for (auto &f : sec_ord_ang_si_) {
            f.add_proj_f_dt(idx, data, dt); ///< add projection of f_dt
            idx += f.get_dim();
        }
    }
}
euler_impl::approx_data::approx_data(generic_dynamics::approx_data &&rhs)
    : generic_dynamics::approx_data(std::move(rhs)),
      NULL_INIT_VECMAP(f_u_),
      NULL_INIT_VECMAP(f_y_lin_off_diag_),
      NULL_INIT_VECMAP(f_x_lin_off_diag_),
      NULL_INIT_VECMAP(f_dt_),
      NULL_INIT_VECMAP(proj_f_u_),
      NULL_INIT_VECMAP(proj_f_x_lin_off_diag_),
      NULL_INIT_VECMAP(proj_f_dt_) {
    // create sparse pattern
    size_t f_st = problem()->get_expr_start(func_);
    auto &dyn = static_cast<const euler_impl &>(func_);
    array_type<size_t, primal_fields> arg_st{};
    arg_st[__x] = problem()->get_expr_start(func_.in_args(__x)[0]);
    arg_st[__y] = problem()->get_expr_start(func_.in_args(__y)[0]);
    arg_st[__u] = problem()->get_expr_start(func_.in_args(__u)[0]);
    size_t dim = func_.dim();
    assert(dim == func_.arg_dim(__x) && dim == func_.arg_dim(__y) &&
           "function dimension must match the dimensions of x and y");
    // setup jacobian
    size_t dim_pos_sec_lin = dyn.n_state_sec_ord_lin_ + dyn.n_state_sec_ord_si_lin_;
    approx_->jac_[__y].insert<sparsity::eye>(f_st, arg_st[__y], dim);
    {
        auto off_diag = approx_->jac_[__y].insert<sparsity::diag>(f_st, arg_st[__y] + dim_pos_sec_lin, dyn.n_state_sec_ord_si_lin_);
        setup_map(f_y_lin_off_diag_, off_diag);
    }
    approx_->jac_[__x].insert<sparsity::diag>(f_st, arg_st[__x], dim).setConstant(-1.0);
    // setup linear offdiag
    {
        size_t ei_off_diag_offset_row = dyn.n_state_sec_ord_si_lin_; // explicit is after implicit
        size_t ei_off_diag_offset_col = ei_off_diag_offset_row + dim_pos_sec_lin;
        auto off_diag = approx_->jac_[__x].insert<sparsity::diag>(f_st + ei_off_diag_offset_row,
                                                                  arg_st[__x] + ei_off_diag_offset_col,
                                                                  dyn.n_state_sec_ord_ang_);
        setup_map(f_x_lin_off_diag_, off_diag);
        dyn_proj_->proj_f_x_.insert<sparsity::diag>(f_st, arg_st[__x], dim).setConstant(-1.0);
        auto proj_off_diag = dyn_proj_->proj_f_x_.insert<sparsity::diag>(f_st,
                                                                         arg_st[__x] + dim_pos_sec_lin,
                                                                         dim_pos_sec_lin);
        setup_map(proj_f_x_lin_off_diag_, proj_off_diag);
    }
    // setup angular offset_diag
    size_t dim_quat_sec_ang = dyn.n_state_sec_ord_ang_ + dyn.n_state_sec_ord_si_ang_;
    if (dim_quat_sec_ang > 0) {
        size_t ang_st = f_st + 2 * dim_pos_sec_lin; // angular starts after linear
        size_t col_offset = 2 * dim_pos_sec_lin + dim_quat_sec_ang;
        {
            for (const auto &ang : dyn.sec_ord_ang_si_) {
                auto f_y_ang = approx_->jac_[__y].insert(ang_st, arg_st[__y] + col_offset, 4, 3, sparsity::dense);
                f_ang_y_off_diag_.emplace_back(f_y_ang.data(), 4, 3);
                auto proj_f_ang_x = dyn_proj_->proj_f_x_.insert(ang_st, arg_st[__y] + col_offset, 4, 3, sparsity::dense);
                proj_f_ang_x_off_diag_.emplace_back(proj_f_ang_x.data(), 4, 3);
                auto proj_f_ang_u = dyn_proj_->proj_f_u_.insert(ang_st, arg_st[__u] + col_offset, 4, 3, sparsity::dense);
                ang_st += ang.dim_q_;
                col_offset += ang.dim_w_;
            }
        }
        {
            for (const auto &ang : dyn.sec_ord_ang_si_) {
                auto f_x_ang = approx_->jac_[__x].insert(ang_st, arg_st[__x] + col_offset, 4, 3, sparsity::dense);
                f_ang_x_off_diag_.emplace_back(f_x_ang.data(), 4, 3);
                auto proj_f_ang_x = dyn_proj_->proj_f_x_.insert(ang_st, arg_st[__x] + col_offset, 4, 3, sparsity::dense);
                proj_f_ang_x_off_diag_.emplace_back(proj_f_ang_x.data(), 4, 3);
                ang_st += ang.dim_q_;
                col_offset += ang.dim_w_;
            }
        }
    }
    {
        size_t dim_pos_sec_all = dim_pos_sec_lin + dim_quat_sec_ang;
        size_t dim_pos_fst_lin = dyn.n_state_fst_ord_lin_;
        auto f_u = approx_->jac_[__u].insert<sparsity::diag>(f_st + dim_pos_sec_all, arg_st[__u], dim_pos_sec_all + dim_pos_fst_lin);
        setup_map(f_u_, f_u);
        auto proj_f_u = dyn_proj_->proj_f_u_.insert<sparsity::diag>(f_st + dim_pos_sec_all, arg_st[__u], f_u_.rows());
        setup_map(proj_f_u_, proj_f_u);
    }
    if (dyn.has_timestep_) {
        auto f_dt = approx_->jac_[__u].insert(f_st, problem()->get_expr_start(dyn.timestep_var_), func_.dim(), 1, sparsity::dense);
        setup_map(f_dt_, f_dt);
        auto proj_f_dt = dyn_proj_->proj_f_u_.insert(f_st, problem()->get_expr_start(dyn.timestep_var_), func_.dim(), 1, sparsity::dense);
        setup_map(proj_f_dt_, proj_f_dt);
    }
}

void euler::add_dt(scalar_t dt) {
    if (dt <= 0.0)
        throw std::runtime_error("Time step must be positive");
    IMPL->dt_ = dt;
    IMPL->has_timestep_ = false;
}
void euler::add_dt(const var &dt) {
    IMPL->timestep_var_ = dt;
    IMPL->has_timestep_ = true;
}

euler::euler(const std::string &name)
    : func(std::make_shared<euler_impl>(name)) {}

} // namespace moto