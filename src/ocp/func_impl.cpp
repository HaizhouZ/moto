#include <condition_variable>
#include <fmt/ranges.h>
#include <moto/core/external_function.hpp>
#include <moto/ocp/impl/func.hpp>
#include <moto/utils/codegen.hpp>
#include <mutex>

#include <moto/ocp/problem.hpp>

namespace moto {

func_approx_data_ptr_t generic_func::create_approx_data(sym_data &primal, merit_data &raw, shared_data &shared) const {
    if (field() - __dyn >= field::num_func)
        throw std::runtime_error(fmt::format("create_approx_data cannot be called for func {} type {}",
                                             name(), field::name(field())));
    if (in_args().empty())
        throw std::runtime_error(fmt::format("in args unset for func {} in field {}",
                                             name(), field::name(field())));
    return std::make_unique<func_approx_data>(primal, raw, shared, *this);
}
void generic_func::compute_approx(func_approx_data &data,
                                  bool eval_val, bool eval_jac, bool eval_hess) const {
    if (eval_val)
        value_impl(data);
    if (eval_jac)
        jacobian_impl(data);
    if (eval_hess)
        hessian_impl(data);
}

void generic_func::value_impl(func_approx_data &data) const {
    try {
        value(data);
    } catch (const std::bad_function_call &ex) {
        throw std::runtime_error(fmt::format("Function {} has no value implementation, please implement it or load from shared library", name()));
    }
}
void generic_func::jacobian_impl(func_approx_data &data) const {
    try {
        jacobian(data);
    } catch (const std::bad_function_call &ex) {
        throw std::runtime_error(fmt::format("Function {} has no jacobian implementation, please implement it or load from shared library", name()));
    }
}
void generic_func::hessian_impl(func_approx_data &data) const {
    try {
        hessian(data);
    } catch (const std::bad_function_call &ex) {
        throw std::runtime_error(fmt::format("Function {} has no hessian implementation, please implement it or load from shared library", name()));
    }
}

void generic_func::load_external_impl(const std::string &path) {
    auto funcs = load_approx(name_, true, order_ >= approx_order::first, order_ >= approx_order::second);
    value = [eval = std::move(funcs[0])](func_approx_data &d) {
        eval.invoke(d.in_arg_data(), d.v_);
    };
    jacobian = [jac = std::move(funcs[1])](func_approx_data &d) {
        jac.invoke(d.in_arg_data(), d.jac_);
    };

    hessian = [hess = std::move(funcs[2])](func_approx_data &d) {
        hess.invoke(d.in_arg_data(), d.merit_hess_);
    };
}

void generic_func::substitute(const sym &arg, const sym &rhs) {
    if (gen_.task_) {
        gen_.task_->sx_output = cs::SX::substitute(gen_.task_->sx_output, arg, rhs);
    }
    auto in_arg_it = std::find_if(in_args_.begin(), in_args_.end(),
                                  [&arg](const sym &s) { return s.uid() == arg.uid(); });
    if (in_arg_it == in_args_.end())
        throw std::runtime_error(fmt::format("func {} substitute failed: argument to replace not found", name_));
    auto in_arg_pos = in_arg_it - in_args_.begin();
    in_args_.at(in_arg_pos) = rhs; // update the in_args_ to point to the new sym
    dep_.at(in_arg_pos) = rhs;     // update the dep_ to point to the new sym
}

void generic_func::set_from_casadi(const var_inarg_list &in_args, const cs::SX &out) {
    add_arguments(in_args);
    if (gen_.task_)
        throw std::runtime_error(fmt::format("func {} already has an expression set", name_));
    else {
        gen_.task_ = new gen_info::task_type();
        gen_.task_->sx_output = out;
    }
}

/**
 * @brief Code generation helper for functions
 *
 */
void make_codegen_task(generic_func *f);

generic_func generic_func::share(bool copy_args, const var_inarg_list &skip_copy_args) const {
    if (!wait_until_ready())
        throw std::runtime_error(fmt::format("func {} not ready, cannot share", name()));
    gen_.copy_task = false;
    generic_func f(*this);
    gen_.copy_task = true;
    f.finalized_ = true;                               // keep finalized
    f.async_ready_status_ = this->async_ready_status_; // share the async status
    if (copy_args) {
        bool skip_copy[in_args_.size()] = {false};
        for (const sym &s : skip_copy_args) {
            if (has_arg(s)) {
                skip_copy[sym_uid_idx_.at(s.uid())] = true;
            }
        }
        size_t idx = 0;
        for (sym &s : in_args_) {
            if (skip_copy[idx++])
                continue;
            auto s_new = s.clone(s.name() + "_copy");
            if (bool(s.dual()) && has_arg(s.dual())) {
                skip_copy[sym_uid_idx_.at(s.dual()->uid())] = true;
            }
            f.substitute(s, s_new);
        }
    }
    return f;
}

void generic_func::finalize_impl() {
    if (gen_.task_ && !gen_.task_->sx_output.is_empty()) {
        // prune unused args
        auto &out = gen_.task_->sx_output;
        std::vector<size_t> unused_args;
        for (const sym &s : in_args_) {
            if (!cs::SX::depends_on(out, s)) {
                unused_args.push_back(s.uid());
            }
        }
        for (size_t uid : unused_args) {
            std::erase_if(in_args_, [uid](const sym &arg) { return arg.uid() == uid; });
            std::erase_if(dep_, [uid](const sym &arg) { return arg.uid() == uid; });
        }
    }
    for (auto &s : in_args_) {
        auto f = s->field();
        if (in_field(f, primal_fields)) {
            arg_dim_[f] += s->dim();
            arg_tdim_[f] += s->tdim();
            arg_num_[f]++;
            arg_by_field_[f].emplace_back(s);
        }
        sym_uid_idx_[s->uid()] = sym_uid_idx_.size();
    }
    if (order_ != approx_order::none && dim_ == dim_tbd && !zero_dim_) {
        throw std::runtime_error(fmt::format("generic_func {} has no dimension set", name_));
    }
    if (!zero_dim_) {
        jac_sp_.assign(in_args_.size(), sparsity::dense);
        // setup default hessian sparsity as @ref default_hess_sp_, which will be later updated by codegen
        hess_sp_.assign(in_args_.size(), std::vector<sparsity>(in_args_.size(), default_hess_sp_));
    }
    if (gen_.task_ && !gen_.task_->sx_output.is_empty()) {
        utils::cs_codegen::task &t = *gen_.task_;
        t.func_name = name_;
        t.sx_inputs = in_args_;
        t.gen_eval = order_ >= approx_order::zero;
        t.gen_jacobian = order_ >= approx_order::first;
        t.gen_hessian = order_ >= approx_order::second;
        t.append_value = field_ == __cost;
        t.append_jac = field_ == __cost;
        t.hess_sp = &hess_sp_;
        t.verbose = false;
        t.force_recompile = false;
        t.keep_generated_src = true;
        // constexpr std::string_view debug_compile_flag = "-g -O0 -march=native";
        // t.jac_compile_flag = debug_compile_flag;
        // t.hess_compile_flag = debug_compile_flag;
        utils::cs_codegen::server::add_job(
            std::move(utils::cs_codegen::generate_and_compile(t)
                          .add_finish_callback([this]() {
                              load_external_impl(); ///< load the generated code
                              set_ready_status(true);
                          })));
    } else {
        set_ready_status(true); ///< set the ready status
    }
    finalized_ = true;
}
void generic_func::setup_ocpwise_info(const ocp *prob) const {
    if (ocpwise_info_map_.contains(prob->uid()))
        return;
    ocpwise_info info;
    for (auto &arg : in_args_) {
        auto f = arg->field();
        if (prob->is_active(arg) && f < field::num_prim) {
            info.arg_by_field_[f].emplace_back(arg);
            info.arg_dim_[f] += arg->dim();
            info.arg_tdim_[f] += arg->tdim();
            info.arg_num_[f]++;
        }
    }
    ocpwise_info_map_[prob->uid()] = info;
}
const var_list &generic_func::active_args(field_t f, const ocp *prob) const {
    field_access_guard(f);
    setup_ocpwise_info(prob);
    return ocpwise_info_map_.at(prob->uid()).arg_by_field_[f];
}
size_t generic_func::active_dim(field_t f, const ocp *prob) const {
    field_access_guard(f);
    setup_ocpwise_info(prob);
    return ocpwise_info_map_.at(prob->uid()).arg_dim_[f];
}
size_t generic_func::active_tdim(field_t f, const ocp *prob) const {
    field_access_guard(f);
    setup_ocpwise_info(prob);
    return ocpwise_info_map_.at(prob->uid()).arg_tdim_[f];
}
size_t generic_func::active_num(field_t f, const ocp *prob) const {
    field_access_guard(f);
    setup_ocpwise_info(prob);
    return ocpwise_info_map_.at(prob->uid()).arg_num_[f];
}
const bool generic_func::check_enable(ocp *prob) const {
    if (disable_if_any_deps_.empty() && enable_if_all_deps_.empty() && enable_if_any_deps_.empty())
        return true;
    bool pass_check = true;
    for (const auto &e : enable_if_any_deps_) {
        if (prob->is_active(e)) {
            pass_check = true;
            goto CHECK_DONE;
        }
    }
    for (const auto &e : disable_if_any_deps_) {
        if (prob->is_active(e)) {
            pass_check = false;
            // fmt::print("func {} disabled because {} active\n", name(), e->name());
            goto CHECK_DONE;
        }
    }
    for (const auto &e : enable_if_all_deps_) {
        if (!prob->is_active(e)) {
            pass_check = false;
            // fmt::print("func {} disabled because {} not active\n", name(), e->name());
            goto CHECK_DONE;
        }
    }
    for (auto &sub_prob : prob->sub_probs()) {
        pass_check &= check_enable(sub_prob.get());
    }
CHECK_DONE:
    return pass_check;
}
void generic_func::enable_if_all(const expr_inarg_list &args) {
    if (enable_if_any_deps_.size() > 0) {
        throw std::runtime_error("Cannot use enable_if_all together with enable_if_any");
    }
    enable_if_all_deps_.insert(enable_if_all_deps_.end(), args.begin(), args.end());
}
void generic_func::disable_if_any(const expr_inarg_list &args) {
    if (enable_if_any_deps_.size() > 0) {
        throw std::runtime_error("Cannot use disable_if_any together with enable_if_any");
    }
    disable_if_any_deps_.insert(disable_if_any_deps_.end(), args.begin(), args.end());
}
void generic_func::enable_if_any(const expr_inarg_list &args) {
    if (enable_if_all_deps_.size() > 0) {
        throw std::runtime_error("Cannot use enable_if_any together with enable_if_all");
    }
    enable_if_any_deps_.insert(enable_if_any_deps_.end(), args.begin(), args.end());
}
generic_func::gen_info::gen_info(const gen_info &rhs) {
    if (rhs.task_ && rhs.copy_task) {
        task_ = new task_type(*rhs.task_);
    }
}
generic_func::gen_info &generic_func::gen_info::operator=(const gen_info &rhs) {
    if (task_) {
        delete task_.get();
        task_ = nullptr;
    }
    if (rhs.task_ && rhs.copy_task) {
        task_ = new task_type(*rhs.task_);
    }
    return *this;
}
generic_func::gen_info::~gen_info() {
    if (task_) {
        delete task_.get();
        task_ = nullptr;
    }
}

void make_codegen_task(generic_func *f) {
}
} // namespace moto