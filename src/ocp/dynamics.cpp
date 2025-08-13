#include <Eigen/LU>
#include <moto/ocp/dynamics.hpp>
#include <moto/ocp/problem.hpp>

namespace moto {

struct dense_dynamics : public generic_dynamics {
    using base = generic_dynamics;

    struct approx_data : public generic_dynamics::approx_data {
        // sparse_mat proj_f_x_;
        // sparse_mat proj_f_u_;
        using aligned_map_t = matrix::AlignedMapType;
        Eigen::PartialPivLU<matrix> lu_; ///< LU decomposition for dense dynamics
        merit_data::dynamics_data *dense_;
        aligned_map_t f_x, f_y;               ///< Jacobian of f_y
        std::vector<aligned_map_t> f_u;       ///< Jacobian of other fields
        aligned_map_t proj_f_x_;              ///< Jacobian of x
        std::vector<aligned_map_t> proj_f_u_; ///< projection of f_u
        approx_data(generic_constr::approx_data &&rhs)
            : generic_dynamics::approx_data(std::move(rhs)),
              f_x(nullptr, 0, 0), f_y(nullptr, 0, 0), proj_f_x_(nullptr, 0, 0) {
            auto &prob = *merit_data_->prob_;
            dense_ = &merit_data_->dynamics_data_;
            size_t f_st = prob.get_expr_start(func_);
            size_t arg_idx = 0;
            auto &in_args = func_.in_args();
            // setup f_y
            auto first_y_arg = *std::find_if(in_args.begin(), in_args.end(), [](const auto &arg) { return arg->field() == __y; });
            auto jac_y = dense_->jac_[__y].insert(f_st, prob.get_expr_start(first_y_arg), func_.dim(), func_.arg_dim(__y), sparsity::dense);
            new (&f_y) aligned_map_t(jac_y.data(), jac_y.rows(), jac_y.cols());
            // setup f_x
            auto first_x_arg = *std::find_if(in_args.begin(), in_args.end(), [](const auto &arg) { return arg->field() == __x; });
            auto jac_x = dense_->jac_[__x].insert(f_st, prob.get_expr_start(first_x_arg), func_.dim(), func_.arg_dim(__x), sparsity::dense);
            new (&f_x) aligned_map_t(jac_x.data(), jac_x.rows(), jac_x.cols());
            // set up projected f_x
            auto f_x = dense_->proj_f_x_.insert(f_st, prob.get_expr_start(first_x_arg), func_.dim(), func_.arg_dim(__x), sparsity::dense);
            new (&proj_f_x_) aligned_map_t(f_x.data(), f_x.rows(), f_x.cols());
            f_u.reserve(func_.arg_num(__u));
            for (auto &arg : in_args) {
                auto f = arg->field();
                if (f < field::num_prim && f != __y && f != __x) {
                    auto m = dense_->jac_[f].insert(f_st, prob.get_expr_start(arg), func_.dim(), arg->dim(), sparsity::dense);
                    new (&jac_[arg_idx]) matrix_ref(m);
                    if (f == __u) {
                        f_u.push_back(aligned_map_t(m.data(), m.rows(), m.cols()));
                        auto p = dense_->proj_f_u_.insert(f_st, prob.get_expr_start(arg), func_.dim(), arg->dim(), sparsity::dense);
                        proj_f_u_.push_back(aligned_map_t(p.data(), p.rows(), p.cols()));
                    }
                } else if (f == __y) {
                    auto cols = f_y.middleCols(prob.get_expr_start(arg), arg->dim());
                    new (&jac_[arg_idx]) matrix_ref(cols);
                } else if (f == __x) {
                    auto cols = f_x.middleCols(prob.get_expr_start(arg), arg->dim());
                    new (&jac_[arg_idx]) matrix_ref(cols);
                }
                arg_idx++;
            }
        }
        // sparse_mat &proj_f_x() override { return proj_f_x_; }
        // sparse_mat &proj_f_u() override { return proj_f_u_; }
    };

    using base::base;

    func_approx_data_ptr_t create_approx_data(sym_data &primal,
                                              merit_data &raw,
                                              shared_data &shared) const override {
        return func_approx_data_ptr_t(make_approx<dense_dynamics>(primal, raw, shared));
    }

    void compute_project_derivatives(func_approx_data &data) override {
        auto &d = data.as<dense_dynamics::approx_data>();
        d.lu_.compute(d.f_y);             // LU decomposition of the dense Jacobian
        d.proj_f_x_ = d.lu_.solve(d.f_x); // Solve for the projection of f_x
        for (size_t i = 0; i < d.f_u.size(); ++i) {
            d.proj_f_u_[i] = d.lu_.solve(d.f_u[i]); // Solve for the projection of f_u
        }
        d.proj_f_res.noalias() = d.lu_.solve(d.dense_->v_); // Solve for the projection of f_res
    }
};

struct explicit_euler {
};

struct semi_implicit_euler {
};

} // namespace moto