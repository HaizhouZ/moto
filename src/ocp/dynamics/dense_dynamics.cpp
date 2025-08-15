#include <Eigen/LU>
#include <moto/ocp/dynamics/dense_dynamics.hpp>
#include <moto/ocp/problem.hpp>

namespace moto {

dense_dynamics::approx_data::~approx_data() {
    if (lu_) {
        delete lu_.get();
    }
}

dense_dynamics::approx_data::approx_data(generic_constr::approx_data &&rhs)
    : generic_dynamics::approx_data(std::move(rhs)),
      f_x_(nullptr, 0, 0), f_y_(nullptr, 0, 0), proj_f_x_(nullptr, 0, 0), lu_(new lu_t()) {
    auto &prob = *merit_data_->prob_;
    size_t f_st = prob.get_expr_start(func_);
    size_t arg_idx = 0;
    auto &in_args = func_.in_args();
    // setup f_y
    auto first_y_arg = *std::find_if(in_args.begin(), in_args.end(), [](const auto &arg) { return arg->field() == __y; });
    auto jac_y = approx_->jac_[__y].insert(f_st, prob.get_expr_start(first_y_arg), func_.dim(), func_.arg_dim(__y), sparsity::dense);
    new (&f_y_) aligned_map_t(jac_y.data(), jac_y.rows(), jac_y.cols());
    // setup f_x
    auto first_x_arg = *std::find_if(in_args.begin(), in_args.end(), [](const auto &arg) { return arg->field() == __x; });
    auto jac_x = approx_->jac_[__x].insert(f_st, prob.get_expr_start(first_x_arg), func_.dim(), func_.arg_dim(__x), sparsity::dense);
    new (&f_x_) aligned_map_t(jac_x.data(), jac_x.rows(), jac_x.cols());
    // set up projected f_x
    auto p_x = dyn_proj_->proj_f_x_.insert(f_st, prob.get_expr_start(first_x_arg), func_.dim(), func_.arg_dim(__x), sparsity::dense);
    new (&proj_f_x_) aligned_map_t(p_x.data(), p_x.rows(), p_x.cols());
    // allocate f_u and proj_f_u_
    f_u_.reserve(func_.arg_num(__u));
    proj_f_u_.reserve(func_.arg_num(__u));
    for (auto &arg : in_args) {
        auto f = arg->field();
        if (f < field::num_prim && f != __y && f != __x) {
            auto m = approx_->jac_[f].insert(f_st, prob.get_expr_start(arg), func_.dim(), arg->dim(), sparsity::dense);
            new (&jac_[arg_idx]) matrix_ref(m);
            if (f == __u) {
                f_u_.push_back(aligned_map_t(m.data(), m.rows(), m.cols()));
                auto p = dyn_proj_->proj_f_u_.insert(f_st, prob.get_expr_start(arg), func_.dim(), arg->dim(), sparsity::dense);
                proj_f_u_.push_back(aligned_map_t(p.data(), p.rows(), p.cols()));
            }
        } else if (f == __y) {
            auto cols = f_y_.middleCols(prob.get_expr_start(arg), arg->dim());
            new (&jac_[arg_idx]) matrix_ref(cols);
        } else if (f == __x) {
            auto cols = f_x_.middleCols(prob.get_expr_start(arg), arg->dim());
            new (&jac_[arg_idx]) matrix_ref(cols);
        }
        arg_idx++;
    }
}

void dense_dynamics::compute_project_derivatives(func_approx_data &data) const {
    auto &d = data.as<dense_dynamics::approx_data>();
    d.lu_->compute(d.f_y_);             // LU decomposition of the dense Jacobian
    d.proj_f_x_ = d.lu_->solve(d.f_x_); // Solve for the projection of f_x
    for (size_t i = 0; i < d.f_u_.size(); ++i) {
        d.proj_f_u_[i] = d.lu_->solve(d.f_u_[i]); // Solve for the projection of f_u
    }
    d.proj_f_res_.noalias() = d.lu_->solve(d.approx_->v_); // Solve for the projection of f_res
}

} // namespace moto