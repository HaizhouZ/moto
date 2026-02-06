// #include <Eigen/LU>
#include <moto/ocp/dynamics/dense_dynamics.hpp>
#include <moto/ocp/problem.hpp>
#include <moto/utils/blasfeo_factorizer/blasfeo_lu.hpp>

namespace moto {

dense_dynamics::approx_data::~approx_data() {
    if (lu_) {
        delete lu_.get();
    }
}

dense_dynamics::approx_data::approx_data(generic_constr::approx_data &&rhs)
    : generic_dynamics::approx_data(std::move(rhs)), lu_(new lu_t()) {
    auto &prob = *merit_data_->prob_;
    size_t f_st = prob.get_expr_start(func_);
    size_t arg_idx = 0;
    auto &in_args = func_.in_args();
    auto &dyn = static_cast<const dense_dynamics &>(func_);
    // setup f_y
    auto &first_y_arg = func_.in_args(__y)[0];
    auto jac_y = approx_->jac_[__y].insert(f_st, prob.get_expr_start_tangent(first_y_arg), func_.dim(), func_.arg_tdim(__y), sparsity::dense);
    f_y_.reset(jac_y);
    // setup f_x
    auto &first_x_arg = func_.in_args(__x)[0];
    auto jac_x = approx_->jac_[__x].insert(f_st, prob.get_expr_start_tangent(first_x_arg), func_.dim(), func_.arg_tdim(__x), sparsity::dense);
    f_x_.reset(jac_x);
    // set up projected f_x
    auto p_x = dyn_proj_->proj_f_x_.insert(f_st, prob.get_expr_start_tangent(first_x_arg), func_.dim(), func_.arg_tdim(__x), sparsity::dense);
    proj_f_x_.reset(p_x);
    // setup f_u_exclusive_ and proj_f_u_exclusive_
    auto &first_u_arg = func_.active_args(__u, &prob)[0];
    auto jac_u = approx_->jac_[__u].insert(f_st, prob.get_expr_start_tangent(first_u_arg), func_.dim(), func_.active_dim(__u, &prob), sparsity::dense);
    f_u_exclusive_.reset(jac_u);
    auto p_u = dyn_proj_->proj_f_u_.insert(f_st, prob.get_expr_start_tangent(first_u_arg), func_.dim(), func_.active_dim(__u, &prob), sparsity::dense);
    proj_f_u_exclusive_.reset(p_u);
    // allocate f_u and proj_f_u_
    f_u_shared_.reserve(func_.active_dim(__u, &prob));
    proj_f_u_shared_.reserve(func_.active_dim(__u, &prob));
    size_t u_col_offset = 0;
    for (const sym &arg : in_args) {
        auto f = arg.field();
        if (prob.is_active(arg))
            if (f == __u) {
                if (dyn.input_shared(arg)) {
                    // create independent jac and proj (and mapping) for shared inputs
                    f_u_shared_.emplace_back(
                        approx_->jac_[__u].insert(f_st, prob.get_expr_start_tangent(arg),
                                                  func_.dim(), arg.tdim(), sparsity::dense));
                    new (&jac_[arg_idx]) matrix_ref(f_u_shared_.back());
                    proj_f_u_shared_.emplace_back(
                        dyn_proj_->proj_f_u_.insert(f_st, prob.get_expr_start_tangent(arg),
                                                    func_.dim(), arg.tdim(), sparsity::dense));

                } else {
                    auto cols = f_u_exclusive_.middleCols(u_col_offset, arg.tdim());
                    new (&jac_[arg_idx]) matrix_ref(cols);
                    u_col_offset += arg.tdim();
                }
            } else if (f == __y) {
                auto cols = f_y_.middleCols(prob.get_expr_start_tangent(arg), arg.tdim());
                new (&jac_[arg_idx]) matrix_ref(cols);
            } else if (f == __x) {
                auto cols = f_x_.middleCols(prob.get_expr_start_tangent(arg), arg.tdim());
                new (&jac_[arg_idx]) matrix_ref(cols);
            }
        arg_idx++;
    }
}

void dense_dynamics::apply_jac_y_inverse_transpose(func_approx_data &data, vector &v, vector &dst) const {
    auto &d = data.as<approx_data>();
    d.lu_->transpose_solve(v, dst);
}

void dense_dynamics::compute_project_derivatives(func_approx_data &data) const {
    auto &d = data.as<approx_data>();
    d.lu_->compute(d.f_y_);                                // LU decomposition of the dense Jacobian
    d.lu_->solve(d.f_x_, d.proj_f_x_);                     // Solve for the projection of f_x
    d.lu_->solve(d.f_u_exclusive_, d.proj_f_u_exclusive_); // Solve for the projection of exclusive f_u
    for (size_t i : range(get_info().num_shared_inputs_)) {
        d.lu_->solve(d.f_u_shared_[i], d.proj_f_u_shared_[i]); // Solve for the projection of shared f_u
    }
    d.lu_->solve(d.approx_->v_, d.proj_f_res_); // Solve for the projection of f_res
}
void dense_dynamics::substitute(const sym &arg, const sym &rhs) {
    generic_dynamics::substitute(arg, rhs);
    // update shared inputs
    if (input_shared(arg)) {
        std::ranges::replace(shared_inputs_, arg, rhs, &var::operator*);
        shared_inputs_indices_.erase(arg.uid());
        shared_inputs_indices_.insert(rhs.uid());
    }
}

void dense_dynamics::finalize_impl() {
    info_.reset(new info(std::move(*info_)));
    // for copying consistency
    get_info().dim_shared_inputs_ = 0;
    // handle reordering of shared inputs
    var_list tmp; // buffer for args to move
    tmp.reserve(shared_inputs_.size());
    for (const sym &s : shared_inputs_) {
        auto it = std::ranges::find(in_args_, s, &var::operator*);
        if (it != in_args_.end()) {
            // move the shared input to the back of the arg list
            tmp.emplace_back(std::move(*it));
        }
    }
    // remove empty ones
    std::erase_if(in_args_, [](auto &&e) { return !e; });
    // append shared inputs at the end
    for (auto &s : tmp) {
        in_args_.emplace_back(std::move(s));
    }
    // base finalization, shared inputs will be synced there
    generic_dynamics::finalize_impl();
    // some might be pruned
    tmp.clear();
    for (var &s : shared_inputs_) {
        if (has_arg(s)) {
            tmp.emplace_back(std::move(s));
            get_info().dim_shared_inputs_ += s->dim();
            shared_inputs_indices_.erase(s->uid());
        }
    }
    shared_inputs_.swap(tmp);
    // set exclusive input counts
    get_info().num_shared_inputs_ = shared_inputs_.size();
    get_info().num_exclusive_inputs_ = generic_func::arg_num(__u) - get_info().num_shared_inputs_;
    get_info().dim_exclusive_inputs_ = generic_func::arg_dim(__u) - get_info().dim_shared_inputs_;
}

void dense_dynamics::mark_shared_inputs(const var_inarg_list &args) {
    for (const sym &arg : args) {
        if (arg.field() == __u) {
            shared_inputs_.push_back(arg);
            shared_inputs_indices_.insert(arg.uid());
        } else
            throw std::runtime_error(
                fmt::format("Only input variables can be shared in dense_dynamics, "
                            "but got sym var {} uid {} of field {}",
                            arg.name(), arg.uid(), arg.field()));
    }
}
} // namespace moto