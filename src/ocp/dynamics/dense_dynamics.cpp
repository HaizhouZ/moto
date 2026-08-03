// #include <Eigen/LU>
#include <moto/ocp/dynamics/dense_dynamics.hpp>
#include <moto/ocp/problem.hpp>
#include <moto/utils/blasfeo_factorizer/blasfeo_lu.hpp>
namespace moto {

dense_dynamics::clone_ptr dense_dynamics::clone() const {
    return new dense_dynamics(*this);
}

dense_dynamics::approx_data::~approx_data() {
    if (lu_) {
        delete lu_.get();
    }
}

dense_dynamics::approx_data::approx_data(generic_constr::approx_data &&rhs)
    : generic_dynamics::approx_data(std::move(rhs)), lu_(new lu_t()) {
    auto &prob = *lag_data_->prob_;
    size_t f_st = prob.get_expr_start(func_);
    size_t arg_idx = 0;
    auto &in_args = func_.in_args();
    // setup f_y
    auto &first_y_arg = func_.in_args(__y)[0];
    auto jac_y = approx_->jac_[__y].insert(f_st, prob.get_expr_start_tangent(first_y_arg), func_.dim(), func_.arg_tdim(__y), sparsity::dense);
    f_y_.reset(jac_y);
    size_t y_col = 0;
    for (const sym &arg : in_args) {
        auto f = arg.field();
        if (prob.is_active(arg))
            if (f == __y) {
                auto cols = f_y_.middleCols(y_col, arg.tdim());
                new (&jac_[arg_idx]) matrix_ref(cols);
                y_col += arg.tdim();
            }
        arg_idx++;
    }
}

void dense_dynamics::apply_jac_y_inverse_transpose(func_approx_data &data,
                                                   vector_ref v,
                                                   vector_ref dst) const {
    auto &d = data.as<approx_data>();
    d.lu_->transpose_solve(v, dst);
}

void dense_dynamics::compute_project_jacobians(func_approx_data &data) const {
    auto &d = data.as<approx_data>();
    d.lu_->compute(d.f_y_);                                // LU decomposition of the dense Jacobian
    d.lu_->solve(d.f_x_, d.proj_f_x_);                     // Solve for the projection of f_x
    if (d.f_u_exclusive_.cols())
        d.lu_->solve(d.f_u_exclusive_, d.proj_f_u_exclusive_); // Solve for the projection of exclusive f_u
    for (size_t i : range(d.f_u_shared_.size())) {
        d.lu_->solve(d.f_u_shared_[i], d.proj_f_u_shared_[i]); // Solve for the projection of shared f_u
    }
}

void dense_dynamics::compute_project_residual(func_approx_data &data) const {
    auto &d = data.as<approx_data>();
    d.lu_->solve(d.v_, d.proj_f_res_); // Solve the function-local residual
}
void dense_dynamics::finalize_impl() {
    disable_jacobian_sparsity_detection();
    generic_dynamics::finalize_impl();
}

} // namespace moto
