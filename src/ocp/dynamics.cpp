#include <Eigen/LU>
#include <moto/ocp/dynamics.hpp>
#include <moto/ocp/problem.hpp>

namespace moto {

struct dense_dynamics : public generic_dynamics {
    using base = generic_dynamics;

    struct approx_data : public generic_dynamics::approx_data {
        sparse_mat proj_f_x_;
        sparse_mat proj_f_u_;
        Eigen::PartialPivLU<matrix> lu_; ///< LU decomposition for dense dynamics
        merit_data::dynamics_data *dense_;
        array_type<matrix, primal_fields> dense_jac_; // jacobian
        approx_data(generic_constr::approx_data &&rhs)
            : generic_dynamics::approx_data(std::move(rhs)) {
            size_t arg_idx[std::size(primal_fields)] = {0, 0, 0}; ///< argument index for x, u, y
            auto &prob = *merit_data_->prob_;
            dense_ = &merit_data_->dynamics_data_;
            size_t f_st = prob.get_expr_start(func_);
            for (auto &arg : func_.in_args()) {
                auto f = arg->field();
                if (f == __x) {
                    dense_->proj_f_x_.insert(f_st, prob.get_expr_start(arg), func_.dim(), arg->dim(), sparsity::dense);
                } else if (f == __u) {
                    dense_->proj_f_u_.insert(f_st, prob.get_expr_start(arg), func_.dim(), arg->dim(), sparsity::dense);
                }
            }
        }
        void setup_jacobian() override {
            auto &prob = *merit_data_->prob_;
            size_t f_st = prob.get_expr_start(func_);
            size_t arg_idx = 0;
            for (auto f : primal_fields) {
                dense_jac_[f].resize(func_.dim(), prob.dim(f));
                dense_jac_[f].setZero();
            }
            for (const sym &arg : func_.in_args()) {
                auto f = arg.field();
                if (f < field::num_prim) {
                    new (&jac_[arg_idx]) matrix_ref(dense_jac_[arg.field()].middleCols(prob.get_expr_start(arg), arg.dim()));
                }
            }
        }
        sparse_mat &proj_f_x() override { return proj_f_x_; }
        sparse_mat &proj_f_u() override { return proj_f_u_; }
    };

    using base::base;

    func_approx_data_ptr_t create_approx_data(sym_data &primal,
                                              merit_data &raw,
                                              shared_data &shared) const override {
        return func_approx_data_ptr_t(make_approx<dense_dynamics>(primal, raw, shared));
    }

    void compute_project_derivatives(func_approx_data &data) override {
        auto &d = data.as<dense_dynamics::approx_data>();
        d.lu_.compute(d.dense_jac_[__y]); // LU decomposition of the dense Jacobian
        for (auto &arg : in_args()) {
            auto f = arg->field();
            // if (f == __x) {
            //     d.proj_f_x_ = d.lu_.solve(d.dense_jac_[__x]); // Solve for the projection of f_x
            // } else if (f == __u) {
            //     d.proj_f_u_ = d.lu_.solve(d.dense_jac_[__u]); // Solve for the projection of f_u
            // }
        }
        d.proj_f_res.noalias() = d.lu_.solve(d.dense_->v_);           // Solve for the projection of f_res
    }
};

struct explicit_euler {
};

struct semi_implicit_euler {
};

} // namespace moto