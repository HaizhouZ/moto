// #include <Eigen/LU>
#include <moto/ocp/dynamics/dense_dynamics.hpp>
#include <moto/ocp/problem.hpp>
#include <moto/utils/blasfeo_factorizer/blasfeo_lu.hpp>
namespace moto {

bool dense_dynamics::input_shared(const sym &s) const {
    return shared_inputs_indices_.contains(s.uid());
}

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
    size_t exclusive_u_dim = 0, shared_u_dim = 0;
    const sym *first_u_arg = nullptr;
    for (const sym &arg : func_.in_args(__u)) {
        if (!prob.is_active(arg))
            continue;
        first_u_arg = first_u_arg == nullptr ? &arg : first_u_arg;
        (dyn.input_shared(arg) ? shared_u_dim : exclusive_u_dim) += arg.tdim();
    }
    if (first_u_arg == nullptr) {
        throw std::runtime_error(fmt::format("dense dynamics {} has no active input argument", func_.name()));
    }
    // setup f_u_exclusive_ and proj_f_u_exclusive_
    auto jac_u = approx_->jac_[__u].insert(f_st, prob.get_expr_start_tangent(*first_u_arg), func_.dim(), exclusive_u_dim, sparsity::dense);
    f_u_exclusive_.reset(jac_u);
    auto p_u = dyn_proj_->proj_f_u_.insert(f_st, prob.get_expr_start_tangent(*first_u_arg), func_.dim(), exclusive_u_dim, sparsity::dense);
    proj_f_u_exclusive_.reset(p_u);
    // allocate f_u and proj_f_u_
    f_u_shared_.reserve(shared_u_dim);
    proj_f_u_shared_.reserve(shared_u_dim);
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

void dense_dynamics::compute_project_jacobians(func_approx_data &data) const {
    auto &d = data.as<approx_data>();
    d.lu_->compute(d.f_y_);                                // LU decomposition of the dense Jacobian
    d.lu_->solve(d.f_x_, d.proj_f_x_);                     // Solve for the projection of f_x
    d.lu_->solve(d.f_u_exclusive_, d.proj_f_u_exclusive_); // Solve for the projection of exclusive f_u
    for (size_t i : range(d.f_u_shared_.size())) {
        d.lu_->solve(d.f_u_shared_[i], d.proj_f_u_shared_[i]); // Solve for the projection of shared f_u
    }
}

void dense_dynamics::compute_project_residual(func_approx_data &data) const {
    auto &d = data.as<approx_data>();
    d.lu_->solve(d.approx_->v_, d.proj_f_res_); // Solve for the projection of f_res
}
void dense_dynamics::substitute(const sym &arg, const sym &rhs) {
    generic_dynamics::substitute(arg, rhs);
    // update shared inputs
    if (input_shared(arg)) {
        std::replace(shared_inputs_.begin(), shared_inputs_.end(), arg, rhs);
        shared_inputs_indices_.erase(arg.uid());
        shared_inputs_indices_.insert(rhs.uid());
    }
}

void dense_dynamics::finalize_impl() {
    disable_jacobian_sparsity_detection();
    // handle reordering of shared inputs
    var_list tmp; // buffer for args to move
    tmp.reserve(shared_inputs_.size());
    for (const sym &s : shared_inputs_) {
        auto it = std::find(in_args_.begin(), in_args_.end(), s);
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
    shared_inputs_indices_.clear();
    for (var &s : shared_inputs_) {
        if (has_arg(s)) {
            tmp.emplace_back(std::move(s));
            shared_inputs_indices_.insert(tmp.back()->uid());
        }
    }
    shared_inputs_.swap(tmp);
}

void dense_dynamics::mark_shared_inputs(const var_inarg_list &args) {
    field_write_guard();
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
