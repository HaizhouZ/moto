#include <moto/ocp/dynamics.hpp>
#include <moto/ocp/problem.hpp>

namespace moto {

generic_dynamics::approx_data::approx_data(base::approx_data &&rhs)
    : approx_data(std::move(rhs), false) {}

generic_dynamics::approx_data::approx_data(base::approx_data &&rhs,
                                           bool sparse_projection)
    : base::approx_data(std::move(rhs)),
      proj_f_res_(problem()->extract(lag_data_->dynamics_data_.proj_f_res_, func_)) {
    approx_ = &lag_data_->approx_[__dyn];
    dyn_proj_ = &lag_data_->dynamics_data_;
    const auto &dyn = static_cast<const generic_dynamics &>(func_);
    auto &prob = *lag_data_->prob_;
    const size_t f_st = prob.get_expr_start(func_);
    const sym &first_x = func_.in_args(__x).front();
    f_x_.reset(approx_->jac_[__x].insert(
        f_st, prob.get_expr_start_tangent(first_x), func_.dim(),
        func_.arg_tdim(__x), sparsity::dense));
    if (!sparse_projection)
        proj_f_x_.reset(dyn_proj_->proj_f_x_.insert(
            f_st, prob.get_expr_start_tangent(first_x), func_.dim(),
            func_.arg_tdim(__x), sparsity::dense));

    size_t exclusive_dim = 0;
    const sym *first_exclusive_u = nullptr;
    for (const sym &arg : func_.in_args(__u)) {
        if (!prob.is_active(arg)) continue;
        if (!dyn.input_shared(arg)) {
            if (!first_exclusive_u) first_exclusive_u = &arg;
            exclusive_dim += arg.tdim();
        }
    }
    if (exclusive_dim) {
        f_u_exclusive_.reset(approx_->jac_[__u].insert(
            f_st, prob.get_expr_start_tangent(*first_exclusive_u), func_.dim(),
            exclusive_dim, sparsity::dense));
        if (!sparse_projection)
            proj_f_u_exclusive_.reset(dyn_proj_->proj_f_u_.insert(
            f_st, prob.get_expr_start_tangent(*first_exclusive_u), func_.dim(),
            exclusive_dim,
            sparsity::dense));
    }

    size_t x_col = 0, u_col = 0;
    for (size_t i = 0; i < func_.in_args().size(); ++i) {
        const sym &arg = func_.in_args(i);
        if (!prob.is_active(arg)) continue;
        if (arg.field() == __x) {
            new (&jac_[i]) matrix_ref(f_x_.middleCols(x_col, arg.tdim()));
            x_col += arg.tdim();
        } else if (arg.field() == __u && dyn.input_shared(arg)) {
            f_u_shared_.emplace_back(approx_->jac_[__u].insert(
                f_st, prob.get_expr_start_tangent(arg), func_.dim(), arg.tdim(),
                sparsity::dense));
            new (&jac_[i]) matrix_ref(f_u_shared_.back());
            if (!sparse_projection)
                proj_f_u_shared_.emplace_back(dyn_proj_->proj_f_u_.insert(
                    f_st, prob.get_expr_start_tangent(arg), func_.dim(), arg.tdim(),
                    sparsity::dense));
        } else if (arg.field() == __u) {
            new (&jac_[i]) matrix_ref(f_u_exclusive_.middleCols(u_col, arg.tdim()));
            u_col += arg.tdim();
        }
    }
}

bool generic_dynamics::input_shared(const sym &s) const {
    return shared_inputs_indices_.contains(s.uid());
}

void generic_dynamics::substitute(const sym &arg, const sym &rhs) {
    generic_constr::substitute(arg, rhs);
    if (!input_shared(arg))
        return;
    std::replace(shared_inputs_.begin(), shared_inputs_.end(), arg, rhs);
    shared_inputs_indices_.erase(arg.uid());
    shared_inputs_indices_.insert(rhs.uid());
}

void generic_dynamics::finalize_impl() {
    var_list reordered;
    reordered.reserve(shared_inputs_.size());
    for (const sym &s : shared_inputs_)
        if (auto it = std::find(in_args_.begin(), in_args_.end(), s); it != in_args_.end())
            reordered.emplace_back(std::move(*it));
    std::erase_if(in_args_, [](const auto &arg) { return !arg; });
    for (auto &arg : reordered)
        in_args_.emplace_back(std::move(arg));
    prepare_dynamics_codegen();
    generic_constr::finalize_impl();
    reordered.clear();
    shared_inputs_indices_.clear();
    for (var &s : shared_inputs_)
        if (has_arg(s)) {
            reordered.emplace_back(std::move(s));
            shared_inputs_indices_.insert(reordered.back()->uid());
        }
    shared_inputs_.swap(reordered);
}

void generic_dynamics::mark_shared_inputs(const var_inarg_list &args) {
    field_write_guard();
    for (const sym &arg : args) {
        if (arg.field() != __u)
            throw std::runtime_error(fmt::format(
                "Only input variables can be shared in dynamics, got {} in field {}",
                arg.name(), arg.field()));
        shared_inputs_.push_back(arg);
        shared_inputs_indices_.insert(arg.uid());
    }
}
} // namespace moto
