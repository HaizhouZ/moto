#include <algorithm>
#include <condition_variable>
#include <fmt/ranges.h>
#include <moto/core/external_function.hpp>
#include <moto/ocp/impl/func.hpp>
#include <moto/utils/codegen.hpp>
#include <mutex>

#include <moto/ocp/problem.hpp>

namespace moto {

namespace {
bool lowering_trace_enabled() {
    static const bool enabled = std::getenv("MOTO_TRACE_COMPOSE") != nullptr;
    return enabled;
}
} // namespace

func_approx_data_ptr_t generic_func::create_approx_data(sym_data &primal, lag_data &raw, shared_data &shared) const {
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

void generic_func::setup_hess() {
}

void generic_func::hessian_impl(func_approx_data &data) const {
    try {
        hessian(data);
    } catch (const std::bad_function_call &ex) {
        throw std::runtime_error(fmt::format("Function {} has no hessian implementation, please implement it or load from shared library", name()));
    }
}

void generic_func::load_external_impl(const std::string &path) {
    const std::string func_name =
        (gen_.task_ && !gen_.task_->func_name.empty()) ? gen_.task_->func_name : name_;
    auto funcs = load_approx(func_name, true, order_ >= approx_order::first, order_ >= approx_order::second);
    value = [eval = std::move(funcs[0])](func_approx_data &d) {
        eval.invoke(d.in_arg_data(), d.v_);
    };
    jacobian = [jac = std::move(funcs[1])](func_approx_data &d) {
        jac.invoke(d.in_arg_data(), d.jac_);
    };

    hessian = [hess = std::move(funcs[2])](func_approx_data &d) {
        hess.invoke(d.in_arg_data(), d.lag_hess_);
    };
    setup_hess();
}

void generic_func::substitute(const sym &arg, const sym &rhs) {
    if (gen_.task_) {
        gen_.task_->sx_output = cs::SX::substitute(gen_.task_->sx_output, arg, rhs);
    }
    auto in_arg_it = std::find(in_args_.begin(), in_args_.end(), arg);
    if (in_arg_it == in_args_.end())
        throw std::runtime_error(fmt::format("func {} substitute failed: argument to replace not found", name_));
    // update the in_args_ to point to the new sym
    *in_arg_it = rhs;
    // update the dep_ to point to the new sym
    std::replace(dep_.begin(), dep_.end(), arg, rhs);
}

bool generic_func::has_u_arg() const {
    return std::any_of(in_args_.begin(), in_args_.end(),
                       [](const sym &arg) { return arg.field() == __u; });
}

bool generic_func::has_pure_x_primal_args() const {
    bool has_x = false;
    for (const sym &arg : in_args_) {
        if (!in_field(arg.field(), primal_fields)) {
            continue;
        }
        if (arg.field() != __x) {
            return false;
        }
        has_x = true;
    }
    return has_x;
}

void generic_func::lower_x_to_y_in_place(std::string_view context, size_t problem_uid) {
    for (const sym &arg : in_args_) {
        if (arg.field() != __x) {
            continue;
        }
        if (lowering_trace_enabled()) {
            if (problem_uid == static_cast<size_t>(-1)) {
                fmt::print("lowering {}: {} -> {} (x -> y)\n",
                           context, arg.name(), arg.next()->name());
            } else {
                fmt::print("lowering {} in composed ocp uid {}: {} -> {} (x -> y)\n",
                           context, problem_uid, arg.name(), arg.next()->name());
            }
        }
        substitute_argument(arg, arg.next());
    }
}

shared_expr generic_func::lower_expr_x_to_y_cached(std::string_view context, size_t problem_uid) {
    if (lowered_) {
        return lowered_;
    }

    shared_expr lowered_expr(clone());
    auto *lowered_func = dynamic_cast<generic_func *>(lowered_expr.get());
    if (lowered_func != nullptr) {
        lowered_func->lower_x_to_y_in_place(context, problem_uid);
    }
    lowered_ = lowered_expr;
    return lowered_;
}

void generic_func::set_from_casadi(const var_inarg_list &in_args, const cs::SX &out) {
    if (gen_.task_)
        throw std::runtime_error(fmt::format("func {} already has a casadi codegen task", name_));
    else {
        add_arguments(in_args);
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
    f.finalized_ = true; // keep finalized
    /// @todo make remapping
    // f.async_ready_status_ = this->async_ready_status_; // share the async status
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
            if (!cs::SX::depends_on(out, s) && !skip_unused_arg_check_.contains(s.uid())) {
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
            info_->arg_dim_[f] += s->dim();
            info_->arg_tdim_[f] += s->tdim();
            info_->arg_num_[f]++;
            info_->arg_by_field_[f].emplace_back(s);
        }
        sym_uid_idx_[s->uid()] = sym_uid_idx_.size();
    }
    if (order_ != approx_order::none && dim_ == dim_tbd && !zero_dim_) {
        throw std::runtime_error(fmt::format("generic_func {} has no dimension set", name_));
    }
    if (!zero_dim_) {
        jac_sp_.clear();
        jac_sp_.reserve(in_args_.size());
        for (const sym &arg : in_args_) {
            jac_sp_.push_back({sparsity::dense, 0, 0, this->dim_, arg.tdim()});
        }
        // setup default hessian sparsity as @ref default_hess_sp_, which can be refined by codegen
        hess_sp_.assign(in_args_.size(), {});
        for (size_t i : range(in_args_.size())) {
            hess_sp_[i].resize(in_args_.size());
            for (size_t j : range(in_args_.size())) {
                hess_sp_[i][j] = {default_hess_sp_, jac_sp_[i].col_offset, jac_sp_[j].col_offset,
                                  jac_sp_[i].cols, jac_sp_[j].cols};
            }
        }
    }
    if (gen_.task_ && !gen_.task_->sx_output.is_empty()) {
        utils::cs_codegen::task &t = *gen_.task_;
        // Function names are already modeled to be stable identifiers.
        // Reusing the same generated symbol name lets finalized clones share
        // the compiled artifact instead of forcing a rebuild per expr uid.
        t.func_name = name_;
        t.sx_inputs = in_args_;
        t.gen_eval = order_ >= approx_order::zero;
        t.gen_jacobian = order_ >= approx_order::first;
        t.gen_hessian = order_ >= approx_order::second;
        t.append_value = field_ == __cost;
        t.append_jac = field_ == __cost;
        // t.jac_sp = detect_jacobian_sparsity_ ? &jac_sp_ : nullptr;
        t.jac_sp = nullptr;
        t.hess_sp = &hess_sp_;
        t.verbose = false;
        t.force_recompile = false;
        t.keep_generated_src = true;
        // constexpr std::string_view debug_compile_flag = "-g -O0 -march=native";
        // t.jac_compile_flag = debug_compile_flag;
        // t.hess_compile_flag = debug_compile_flag;
        auto jobs = std::move(utils::cs_codegen::generate_and_compile(t)
                                  .add_finish_callback([this]() {
                                      load_external_impl(); ///< load the generated code
                                      set_ready_status(true);
                                  }));
        if (std::getenv("MOTO_SYNC_CODEGEN") != nullptr) {
            jobs.wait_until_finished();
        } else {
            utils::cs_codegen::server::add_job(std::move(jobs));
        }
    } else {
        set_ready_status(true); ///< set the ready status
    }
    finalized_ = true;
}
bool generic_func::setup_ocpwise_info(const ocp_base *prob) const {
    if (ocpwise_info_map_->contains(prob->uid()))
        return false;
    auto &tmp = *ocpwise_info_map_->insert(
                                      {prob->uid(),
                                       info::holder(std::make_unique<info>())})
                     .first->second;
    // fmt::println("Setting up ocpwise info for func {} in prob uid {}", name(), prob->uid());
    // setup
    for (auto &arg : in_args_) {
        auto f = arg->field();
        if (prob->is_active(arg) && f < field::num_prim) {
            tmp.arg_by_field_[f].emplace_back(arg);
            tmp.arg_dim_[f] += arg->dim();
            tmp.arg_tdim_[f] += arg->tdim();
            tmp.arg_num_[f]++;
        }
    }
    // fmt::println(" - active args by field:");
    // for (size_t f : primal_fields) {
    //     fmt::println("   - field {}: num {}, dim {}, tdim {}, args {}",
    //                  f, tmp.arg_num_[f], tmp.arg_dim_[f], tmp.arg_tdim_[f], tmp.arg_by_field_[f]);
    // }
    return true;
}
const var_list &generic_func::active_args(field_t f, const ocp_base *prob) const {
    field_read_guard(f);
    setup_ocpwise_info(prob);
    return ocpwise_info_map_->at(prob->uid())->arg_by_field_[f];
}
size_t generic_func::active_dim(field_t f, const ocp_base *prob) const {
    field_read_guard(f);
    setup_ocpwise_info(prob);
    return ocpwise_info_map_->at(prob->uid())->arg_dim_[f];
}
size_t generic_func::active_tdim(field_t f, const ocp_base *prob) const {
    field_read_guard(f);
    setup_ocpwise_info(prob);
    return ocpwise_info_map_->at(prob->uid())->arg_tdim_[f];
}
size_t generic_func::active_num(field_t f, const ocp_base *prob) const {
    field_read_guard(f);
    setup_ocpwise_info(prob);
    return ocpwise_info_map_->at(prob->uid())->arg_num_[f];
}
const bool generic_func::check_enable(ocp_base *prob) const {
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
    field_write_guard();
    if (enable_if_any_deps_.size() > 0) {
        throw std::runtime_error("Cannot use enable_if_all together with enable_if_any");
    }
    enable_if_all_deps_.insert(enable_if_all_deps_.end(), args.begin(), args.end());
}
void generic_func::disable_if_any(const expr_inarg_list &args) {
    field_write_guard();
    if (enable_if_any_deps_.size() > 0) {
        throw std::runtime_error("Cannot use disable_if_any together with enable_if_any");
    }
    disable_if_any_deps_.insert(disable_if_any_deps_.end(), args.begin(), args.end());
}
void generic_func::enable_if_any(const expr_inarg_list &args) {
    field_write_guard();
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
