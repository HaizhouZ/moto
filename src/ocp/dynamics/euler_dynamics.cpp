#include <Eigen/Geometry>
#include <moto/ocp/dynamics/euler_dynamics.hpp>
#include <moto/ocp/problem.hpp>
#include <moto/utils/optional_boolean.hpp>

namespace moto {

struct dynamics_with_dt {
    size_t pos_idx_ = 0; ///< position offset in the full state
    size_t vel_idx_ = 0; ///< velocity offset in the full state

    var_list pos_;   ///< position variables
    var_list vel_;   ///< velocity variables
    var_list input_; ///< acceleration variables

    size_t dim_ = 0;
    size_t dim_pos_ = 0;
    size_t dim_vel_ = 0;
    size_t dim_input_ = 0;

    size_t local_idx = 0; ///< local index in angular dynamics

    dynamics_with_dt(var_list &&pos, var_list &&vel, var_list &&input)
        : pos_(std::move(pos)), vel_(std::move(vel)), input_(std::move(input)) {
        for (const auto &v : pos_)
            dim_pos_ += v->dim();
        for (const auto &v : vel_)
            dim_vel_ += v->dim();
        for (const auto &v : input_)
            dim_input_ += v->dim();
        dim_pos_ /= 2; // each position variable has a next position
        dim_vel_ /= 2; // each velocity variable has a next velocity
        dim_ = dim_pos_ + dim_vel_;
    }

    size_t dim_pos() const { return dim_pos_; }     ///< get dimension of position variables
    size_t dim_vel() const { return dim_vel_; }     ///< get dimension of velocity variables
    size_t dim_input() const { return dim_input_; } ///< get dimension of acceleration variables

    virtual void value(func_approx_data &data, scalar_t dt) const = 0; ///< compute value at time step
    virtual void f_dt(func_approx_data &data, vector_ref f_dt_, scalar_t dt) const = 0;
    virtual void add_proj_f_dt(func_approx_data &data, scalar_t dt) const {}
    virtual void calc_proj_jac(func_approx_data &data, scalar_t dt) const {}
    size_t get_dim() const { return dim_; }; ///< get dimension of the state
};

class euler_impl : public generic_dynamics {
  public:
    struct approx_data : public generic_dynamics::approx_data {
        aligned_vector_map_t f_y_lin_off_diag_, f_x_lin_off_diag_, proj_f_x_lin_off_diag_;
        using quat_transform_map_t = Eigen::Matrix<scalar_t, 4, 3>::AlignedMapType;
        std::vector<quat_transform_map_t> f_ang_y_off_diag_, f_ang_x_off_diag_;
        std::vector<quat_transform_map_t> proj_f_ang_x_off_diag_, proj_f_ang_x_off_diag_si_, proj_f_ang_u_off_diag_;
        aligned_vector_map_t f_u_, proj_f_u_, proj_f_u_lin_si_off_diag_;
        aligned_vector_map_t f_dt_, proj_f_dt_;
        approx_data(generic_dynamics::approx_data &&rhs);
    };

  private:
    bool has_timestep_ = false; ///< true if timestep variable is present
    scalar_t dt_ = 0.0;         ///< time step
    var timestep_var_;          ///< time variable
    scalar_t get_dt(func_approx_data &data) const { return has_timestep_ ? data[timestep_var_](0) : dt_; }
    struct list_of_dyn : std::vector<std::unique_ptr<dynamics_with_dt>> {
        size_t dim_pos_ = 0;
        bool finalized_ = false; ///< true if the list is finalized
        void fianlize() {
            if (!finalized_) {
                for (auto &d : *this)
                    dim_pos_ += d->dim_pos();
                finalized_ = true;
            }
        }
    };
    template <bool semi_implicit>
    struct sec_ord_ang : public dynamics_with_dt {
        sym &quat_x_; ///< quaternion cur
        sym &quat_y_; ///< quaternion next
        sym &w_x_;    ///< angular velocity cur
        sym &w_y_;    ///< angular velocity next
        sym &a_u_;    ///< angular acceleration input
        sec_ord_ang(const var &q, const var &qn, const var &w, const var &wn, const var &a)
            : quat_x_(q), quat_y_(qn), w_x_(w), w_y_(wn), a_u_(a),
              dynamics_with_dt({q, qn}, {w, wn}, {a}) {
        }
        void value(func_approx_data &data, scalar_t dt) const override {
            data[quat_y_].normalize(); ///< normalize quaternion
            data[quat_x_].normalize(); ///< normalize quaternion
            auto &d = data.as<approx_data>();
            if constexpr (semi_implicit) {
                set_transform(data, d.f_ang_y_off_diag_[local_idx], dt);
                data.v_.segment(pos_idx_, dim_pos()) = data[quat_y_] - data[quat_x_] - d.f_ang_y_off_diag_[local_idx] * data[w_y_];
            } else {
                set_transform(data, d.f_ang_x_off_diag_[local_idx], dt);
                data.v_.segment(pos_idx_, dim_pos()) = data[quat_y_] - data[quat_x_] + d.f_ang_x_off_diag_[local_idx] * data[w_x_];
            }
            data.v_.segment(vel_idx_, dim_vel()) = data[w_y_] - data[w_x_] - data[a_u_] * dt;
        }
        void f_dt(func_approx_data &data, vector_ref f_dt_, scalar_t dt) const override {
            auto &d = data.as<approx_data>();
            if constexpr (semi_implicit) {
                f_dt_.segment(pos_idx_, dim_pos()) = -d.f_ang_y_off_diag_[local_idx] * data[w_y_] / dt;
            } else {
                f_dt_.segment(pos_idx_, dim_pos()) = d.f_ang_x_off_diag_[local_idx] * data[w_x_] / dt;
            }
            f_dt_.segment(vel_idx_, dim_vel()) = -data[a_u_];
        } ///< compute f_dt for this dynamics
        void add_proj_f_dt(func_approx_data &data, scalar_t dt) const override {
            assert(semi_implicit && "semi-implicit dynamics only");
            auto &d = data.as<approx_data>();
            d.proj_f_dt_.segment(pos_idx_, dim_pos()).noalias() -= d.f_ang_y_off_diag_[local_idx] * d.proj_f_dt_.segment(vel_idx_, dim_vel());
        }
        void set_transform(func_approx_data &data, approx_data::quat_transform_map_t &H_local, scalar_t dt) const {
            Eigen::Quaternion<scalar_t> q(data[quat_y_].data());
            // using transposed_map_t = Eigen::Matrix<scalar_t, 3, 4, Eigen::AutoAlign | Eigen::RowMajor>::MapType;
            // transposed_map_t H_local(H_tr.data(), 3, 4);
            H_local.transpose() << -q.x(), q.w(), q.z(), -q.y(),
                -q.y(), -q.z(), q.w(), q.x(),
                -q.z(), q.y(), -q.x(), q.w();
            if constexpr (semi_implicit) {
                H_local *= 0.5 * dt; ///< semi-implicit euler has a factor of 0.5
            } else {
                H_local *= -0.5 * dt; ///< explicit euler has a factor of -0.5
            }
        }
        void calc_proj_jac(func_approx_data &data, scalar_t dt) const override {
            auto &d = data.as<approx_data>();
            if constexpr (semi_implicit) {
                d.proj_f_ang_x_off_diag_si_[local_idx] = d.f_ang_y_off_diag_[local_idx];
                d.proj_f_ang_u_off_diag_[local_idx] = d.f_ang_y_off_diag_[local_idx] * dt;
            } else {
                d.proj_f_ang_x_off_diag_[local_idx] = d.f_ang_x_off_diag_[local_idx];
            }
        }
    };
    list_of_dyn sec_ord_ang_;    ///< second order angular variables
    list_of_dyn sec_ord_ang_si_; ///< second order angular variables (semi-implicit)
    template <bool semi_implicit>
    struct sec_ord_lin : public dynamics_with_dt {
        sym &pos_x_; ///< position variables
        sym &pos_y_; ///< position variables
        sym &vel_x_; ///< velocity variables
        sym &vel_y_; ///< velocity variables
        sym &acc_u_; ///< acceleration variables
        sec_ord_lin(const var &r, const var &rn, const var &v, const var &vn, const var &a)
            : pos_x_(r), pos_y_(rn), vel_x_(v), vel_y_(vn), acc_u_(a),
              dynamics_with_dt({r, rn}, {v, vn}, {a}) {
            assert(dim_pos() == dim_vel() && "position and velocity dimensions must match");
        }
        void value(func_approx_data &data, scalar_t dt) const override {
            if constexpr (semi_implicit)
                data.v_.segment(pos_idx_, dim_pos()).noalias() = data[pos_y_] - data[pos_x_] - data[vel_y_] * dt;
            else
                data.v_.segment(pos_idx_, dim_pos()).noalias() = data[pos_y_] - data[pos_x_] - data[vel_x_] * dt;
            data.v_.segment(vel_idx_, dim_vel()).noalias() = data[vel_y_] - data[vel_x_] - data[acc_u_] * dt;
        }
        void f_dt(func_approx_data &data, vector_ref f_dt_, scalar_t dt) const override {
            if constexpr (semi_implicit)
                f_dt_.segment(pos_idx_, dim_pos()) = -data[vel_y_];
            else
                f_dt_.segment(pos_idx_, dim_pos()) = -data[vel_x_];
            f_dt_.segment(vel_idx_, dim_vel()) = -data[acc_u_];
        } ///< compute f_dt for this dynamics
        void add_proj_f_dt(func_approx_data &data, scalar_t dt) const override {
            assert(semi_implicit && "semi-implicit dynamics only");
            auto &d = data.as<approx_data>();
            d.proj_f_dt_.segment(pos_idx_, dim_pos()).noalias() -= dt * d.proj_f_dt_.segment(vel_idx_, dim_vel()); ///< semi-implicit euler
        }
    };
    list_of_dyn sec_ord_lin_si_; ///< second order variables semi-implicit
    list_of_dyn sec_ord_lin_;    ///< second order variables

    struct fst_ord_lin : public dynamics_with_dt {
        sym &pos_x_; ///< position variables in state field
        sym &pos_y_; ///< position variables in state field
        sym &vel_u_; ///< velocity variables in control field
        fst_ord_lin(const var &r, const var &rn, const var &v)
            : pos_x_(r), pos_y_(rn), vel_u_(v), dynamics_with_dt({r, rn}, {}, {v}) {
        }
        void value(func_approx_data &data, scalar_t dt) const override {
            data.v_.segment(pos_idx_, dim_pos()) = data[pos_y_] - data[pos_x_] - data[vel_u_] * dt;
        }
        void f_dt(func_approx_data &data, vector_ref f_dt_, scalar_t dt) const override {
            f_dt_.segment(pos_idx_, dim_pos()) = -data[vel_u_];
        } ///< compute f_dt for this dynamics
    };
    list_of_dyn fst_ord_lin_; ///< first order variables

    size_t n_dyn_ = 0;                        ///< number of dynamics variables
    std::vector<dynamics_with_dt *> all_dyn_; ///< all dynamics variables for iteration

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

packed_retval<var, 5> euler::create_2nd_ord_ang(const std::string &name, bool semi_implicit) {
    auto [q, qn] = sym::states(name, 4);
    auto [w, wn] = sym::states(name + "_ang_vel", 3);
    auto a = sym::inputs(name + "_ang_acc", 3);
    auto &dst_lst = semi_implicit ? IMPL->sec_ord_ang_si_ : IMPL->sec_ord_ang_;
    auto &new_ = semi_implicit ? dst_lst.emplace_back(new euler_impl::sec_ord_ang<true>(q, qn, w, wn, a))
                               : dst_lst.emplace_back(new euler_impl::sec_ord_ang<false>(q, qn, w, wn, a));
    new_->local_idx = dst_lst.size() - 1;
    IMPL->n_dyn_++;

    return {std::move(q), std::move(qn), std::move(w), std::move(wn), std::move(a)};
}

packed_retval<var, 5> euler::create_2nd_ord_lin(const std::string &name, size_t dim, bool semi_implicit) {
    auto [r, rn] = sym::states(name, dim);
    auto [v, vn] = sym::states(name + "_vel", dim);
    auto a = sym::inputs(name + "_acc", dim);
    auto &dst_lst = semi_implicit ? IMPL->sec_ord_lin_si_ : IMPL->sec_ord_lin_;
    if (semi_implicit) {
        dst_lst.emplace_back(new euler_impl::sec_ord_lin<true>(r, rn, v, vn, a));
    } else {
        dst_lst.emplace_back(new euler_impl::sec_ord_lin<false>(r, rn, v, vn, a));
    }
    IMPL->n_dyn_++;
    return {std::move(r), std::move(rn), std::move(v), std::move(vn), std::move(a)};
}

packed_retval<var, 3> euler::create_1st_ord_lin(const std::string &name, size_t dim) {
    auto [r, rn] = sym::states(name, dim);
    auto v = sym::inputs(name + "_vel", dim);
    IMPL->fst_ord_lin_.emplace_back(new euler_impl::fst_ord_lin(r, rn, v));
    IMPL->n_dyn_++;
    return {std::move(r), std::move(rn), std::move(v)};
}

void euler_impl::finalize_impl() {
    // make sure the order is right!
    // add positional variables
    size_t s_idx = 0;
    auto inc_s_idx = [&s_idx](size_t &idx, size_t dim) {
        idx = s_idx;
        s_idx += dim;
    };
    using add_type = size_t;
    constexpr static add_type input = 0x1;
    constexpr static add_type pos = 0x2;
    constexpr static add_type vel = 0x4;
    all_dyn_.reserve(n_dyn_);
    auto for_each = [&, this]<typename T>(std::initializer_list<T *> ds_list, add_type flag) {
        for (auto *ds : ds_list) {
            if constexpr (std::is_same_v<T, list_of_dyn>)
                ds->fianlize();
            for (auto &d : *ds) {
                if (flag & pos) {
                    add_arguments(d->pos_);
                    inc_s_idx(d->pos_idx_, d->dim_pos());
                    if constexpr (std::is_pointer_v<typename T::value_type>)
                        all_dyn_.emplace_back(d); ///< add to all dynamics
                    else
                        all_dyn_.emplace_back(d.get()); ///< add to all dynamics
                }
                if (flag & vel) {
                    add_arguments(d->vel_);
                    inc_s_idx(d->vel_idx_, d->dim_vel());
                }
                if (flag & input) {
                    add_arguments(d->input_);
                }
            }
        }
    };
    for_each({&sec_ord_lin_si_, &sec_ord_lin_, &sec_ord_ang_si_, &sec_ord_ang_}, pos);
    for_each({&sec_ord_lin_si_, &sec_ord_lin_, &sec_ord_ang_si_, &sec_ord_ang_}, vel);
    for_each({&fst_ord_lin_}, pos);

    // add acceleration variables
    for_each({&all_dyn_}, input);

    dim_ = 0;
    for (dynamics_with_dt *d : all_dyn_) {
        dim_ += d->get_dim();
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
    for (dynamics_with_dt *d : all_dyn_) {
        d->value(data, dt); ///< compute value at time step
        idx += d->get_dim();
    }
}
void euler_impl::jacobian_impl(func_approx_data &data) const {
    auto &d = data.as<approx_data>();
    scalar_t dt = get_dt(data);
    d.f_u_.setConstant(-dt);
    d.f_y_lin_off_diag_.setConstant(dt);
    d.f_x_lin_off_diag_.setConstant(-dt);
    if (has_timestep_) {
        for (auto *f : all_dyn_) {
            f->f_dt(data, d.f_dt_, dt); ///< compute f_dt for this dynamics
        }
    }
}
void euler_impl::compute_project_derivatives(func_approx_data &data) const {
    auto &d = data.as<approx_data>();
    scalar_t dt = get_dt(data);
    if (sec_ord_lin_si_.dim_pos_)
        d.proj_f_x_lin_off_diag_.head(sec_ord_lin_si_.dim_pos_).setConstant(dt);
    if (sec_ord_lin_.dim_pos_)
        d.proj_f_x_lin_off_diag_.tail(sec_ord_lin_.dim_pos_).setConstant(-dt);
    d.proj_f_u_.setConstant(-dt);
    // d.proj_f_res_ = d.v_;
    /// @todo ang and semi-impliciit res
    if (sec_ord_lin_si_.dim_pos_)
        d.proj_f_u_lin_si_off_diag_.setConstant(dt * dt);
    for (auto *fs : {&sec_ord_ang_si_, &sec_ord_ang_})
        for (auto &f : *fs) {
            f->calc_proj_jac(data, dt); ///< compute projection jacobian
        }

    if (has_timestep_) {
        d.proj_f_dt_ = d.f_dt_;
        // correct semi-implicit dynamics
        for (auto *fs : {&sec_ord_lin_si_, &sec_ord_ang_si_})
            for (auto &f : *fs) {
                f->add_proj_f_dt(data, dt); ///< add projection of f_dt
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
      NULL_INIT_VECMAP(proj_f_u_lin_si_off_diag_),
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
    size_t dim_pos_sec_lin = dyn.sec_ord_lin_.dim_pos_ + dyn.sec_ord_lin_si_.dim_pos_;
    size_t dim_quat_sec_ang = dyn.sec_ord_ang_.dim_pos_ + dyn.sec_ord_ang_si_.dim_pos_;
    size_t dim_pos_sec_all = dim_pos_sec_lin + dim_quat_sec_ang;
    size_t dim_vel_sec_all = dim_pos_sec_lin + std::max<int>((int)dim_quat_sec_ang * 3 / 4, 0);

    approx_->jac_[__y].insert<sparsity::eye>(f_st, arg_st[__y], dim);
    if (dyn.sec_ord_lin_si_.dim_pos_ > 0) {
        auto off_diag = approx_->jac_[__y].insert<sparsity::diag>(f_st, arg_st[__y] + dim_pos_sec_all, dyn.sec_ord_lin_si_.dim_pos_);
        setup_map(f_y_lin_off_diag_, off_diag);
        auto proj_f_u_off_diag = dyn_proj_->proj_f_u_.insert<sparsity::diag>(f_st, arg_st[__u], off_diag.rows());
        setup_map(proj_f_u_lin_si_off_diag_, proj_f_u_off_diag);
    }
    approx_->jac_[__x].insert<sparsity::diag>(f_st, arg_st[__x], dim).setConstant(-1.0);
    dyn_proj_->proj_f_x_.insert<sparsity::diag>(f_st, arg_st[__x], dim).setConstant(-1.0);
    auto proj_off_diag = dyn_proj_->proj_f_x_.insert<sparsity::diag>(f_st,
                                                                     arg_st[__x] + dim_pos_sec_all,
                                                                     dim_pos_sec_lin);
    setup_map(proj_f_x_lin_off_diag_, proj_off_diag);
    // setup linear offdiag
    if (dim_pos_sec_lin > 0) {
        if (dyn.sec_ord_lin_.dim_pos_ > 0) {
            size_t ei_off_diag_offset_row = dyn.sec_ord_lin_si_.dim_pos_; // explicit is after implicit
            size_t ei_off_diag_offset_col = ei_off_diag_offset_row + dim_pos_sec_all;
            auto off_diag = approx_->jac_[__x].insert<sparsity::diag>(f_st + ei_off_diag_offset_row,
                                                                      arg_st[__x] + ei_off_diag_offset_col,
                                                                      dyn.sec_ord_lin_.dim_pos_);
            setup_map(f_x_lin_off_diag_, off_diag);
        }
    }
    // setup angular offset_diag
    if (dim_quat_sec_ang > 0) {
        size_t ang_st = f_st + dim_pos_sec_lin;                // angular starts after linear pos
        size_t col_offset = dim_pos_sec_all + dim_pos_sec_lin; // after pos and line vel
        size_t u_col_offset = dim_pos_sec_lin;                 // after linear vel
        {
            for (const auto &ang : dyn.sec_ord_ang_si_) {
                auto f_y_ang = approx_->jac_[__y].insert(ang_st, arg_st[__y] + col_offset, 4, 3, sparsity::dense);
                f_ang_y_off_diag_.emplace_back(f_y_ang.data(), 4, 3);
                auto proj_f_ang_x = dyn_proj_->proj_f_x_.insert(ang_st, arg_st[__y] + col_offset, 4, 3, sparsity::dense);
                proj_f_ang_x_off_diag_si_.emplace_back(proj_f_ang_x.data(), 4, 3);
                auto proj_f_ang_u = dyn_proj_->proj_f_u_.insert(ang_st, u_col_offset, 4, 3, sparsity::dense);
                proj_f_ang_u_off_diag_.emplace_back(proj_f_ang_u.data(), 4, 3);
                ang_st += ang->dim_pos();
                col_offset += ang->dim_vel();
                u_col_offset += ang->dim_vel();
            }
        }
        {
            for (const auto &ang : dyn.sec_ord_ang_) {
                auto f_x_ang = approx_->jac_[__x].insert(ang_st, arg_st[__x] + col_offset, 4, 3, sparsity::dense);
                f_ang_x_off_diag_.emplace_back(f_x_ang.data(), 4, 3);
                auto proj_f_ang_x = dyn_proj_->proj_f_x_.insert(ang_st, arg_st[__x] + col_offset, 4, 3, sparsity::dense);
                proj_f_ang_x_off_diag_.emplace_back(proj_f_ang_x.data(), 4, 3);
                ang_st += ang->dim_pos();
                col_offset += ang->dim_vel();
            }
        }
    }
    {
        size_t dim_pos_fst_lin = dyn.fst_ord_lin_.dim_pos_;
        auto f_u = approx_->jac_[__u].insert<sparsity::diag>(f_st + dim_pos_sec_all, arg_st[__u], dim_vel_sec_all + dim_pos_fst_lin);
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