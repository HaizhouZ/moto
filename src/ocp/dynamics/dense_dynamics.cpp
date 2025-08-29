// #include <Eigen/LU>
#include <moto/ocp/dynamics/dense_dynamics.hpp>
#include <moto/ocp/problem.hpp>
#include <moto/utils/blasfeo_factorizer/blasfeo_lu.hpp>

namespace moto {

dense_dynamics::dense_dynamics(const std::string &name, approx_order order, size_t dim) : func(impl(name, order, dim, __dyn)) {}
dense_dynamics::dense_dynamics(const std::string &name, const var_inarg_list &in_args, const cs::SX &out,
                               approx_order order) : func(impl(name, in_args, out, order, __dyn)) {}

dense_dynamics::impl::approx_data::~approx_data() {
    if (lu_) {
        delete lu_.get();
    }
}

dense_dynamics::impl::approx_data::approx_data(generic_constr::approx_data &&rhs)
    : generic_dynamics::approx_data(std::move(rhs)),
      f_x_(nullptr, 0, 0), f_y_(nullptr, 0, 0), proj_f_x_(nullptr, 0, 0), NULL_INIT_MAP(f_u_all_), NULL_INIT_MAP(proj_f_u_all_), lu_(new lu_t()) {
    auto &prob = *merit_data_->prob_;
    size_t f_st = prob.get_expr_start(func_);
    size_t arg_idx = 0;
    auto &in_args = func_.in_args();
    // setup f_y
    auto &first_y_arg = func_.in_args(__y)[0];
    auto jac_y = approx_->jac_[__y].insert(f_st, prob.get_expr_start_tangent(first_y_arg), func_.dim(), func_.arg_tdim(__y), sparsity::dense);
    setup_map(f_y_, jac_y);
    // setup f_x
    auto &first_x_arg = func_.in_args(__x)[0];
    auto jac_x = approx_->jac_[__x].insert(f_st, prob.get_expr_start_tangent(first_x_arg), func_.dim(), func_.arg_tdim(__x), sparsity::dense);
    setup_map(f_x_, jac_x);
    // set up projected f_x
    auto p_x = dyn_proj_->proj_f_x_.insert(f_st, prob.get_expr_start_tangent(first_x_arg), func_.dim(), func_.arg_tdim(__x), sparsity::dense);
    setup_map(proj_f_x_, p_x);
    // setup f_u_all
    auto &first_u_arg = func_.in_args(__u)[0];
    auto jac_u = approx_->jac_[__u].insert(f_st, prob.get_expr_start_tangent(first_u_arg), func_.dim(), func_.arg_tdim(__u), sparsity::dense);
    setup_map(f_u_all_, jac_u);
    auto p_u = dyn_proj_->proj_f_u_.insert(f_st, prob.get_expr_start_tangent(first_u_arg), func_.dim(), func_.arg_tdim(__u), sparsity::dense);
    setup_map(proj_f_u_all_, p_u);
    // allocate f_u and proj_f_u_
    f_u_.reserve(func_.arg_num(__u));
    proj_f_u_.reserve(func_.arg_num(__u));
    size_t u_col_offset = 0;
    for (auto &arg : in_args) {
        auto f = arg->field();
        if (f < field::num_prim && f != __y && f != __x) {
            // auto m = approx_->jac_[f].insert(f_st, prob.get_expr_start(arg), func_.dim(), arg->dim(), sparsity::dense);
            // new (&jac_[arg_idx]) matrix_ref(m);
            auto cols = f_u_all_.middleCols(u_col_offset, arg->tdim());
            new (&jac_[arg_idx]) matrix_ref(cols);
            u_col_offset += arg->tdim();
            // if (f == __u) {
            //     // f_u_.push_back(aligned_map_t(m.data(), m.rows(), m.cols()));
            //     auto p = dyn_proj_->proj_f_u_.insert(f_st, prob.get_expr_start(arg), func_.dim(), arg->dim(), sparsity::dense);
            //     proj_f_u_.push_back(aligned_map_t(p.data(), p.rows(), p.cols()));
            // }
        } else if (f == __y) {
            auto cols = f_y_.middleCols(prob.get_expr_start_tangent(arg), arg->tdim());
            new (&jac_[arg_idx]) matrix_ref(cols);
        } else if (f == __x) {
            auto cols = f_x_.middleCols(prob.get_expr_start_tangent(arg), arg->tdim());
            new (&jac_[arg_idx]) matrix_ref(cols);
        }
        arg_idx++;
    }
}

void dense_dynamics::impl::apply_jac_y_inverse_transpose(func_approx_data &data, vector& v, vector& dst) const {
    auto &d = data.as<approx_data>();
    d.lu_->transpose_solve(v, dst);
}

void dense_dynamics::impl::compute_project_derivatives(func_approx_data &data) const {
    auto &d = data.as<approx_data>();
    d.lu_->compute(d.f_y_);             // LU decomposition of the dense Jacobian
    d.lu_->solve(d.f_x_, d.proj_f_x_); // Solve for the projection of f_x
    // for (size_t i = 0; i < d.f_u_.size(); ++i) {
    //     d.proj_f_u_[i] = d.lu_->solve(d.f_u_[i]); // Solve for the projection of f_u
    // }
    d.lu_->solve(d.f_u_all_, d.proj_f_u_all_); // Solve for the projection of f_u_all
    d.lu_->solve(d.approx_->v_, d.proj_f_res_); // Solve for the projection of f_res
}

} // namespace moto