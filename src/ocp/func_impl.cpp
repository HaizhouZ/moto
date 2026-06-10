#include <algorithm>
#include <cstdlib>
#include <map>
#include <mutex>
#include <moto/core/external_function.hpp>
#include <moto/ocp/impl/custom_func.hpp>
#include <moto/ocp/impl/func.hpp>
#include <moto/utils/codegen.hpp>

#include <moto/ocp/problem.hpp>

namespace moto {

namespace {
bool lowering_trace_enabled() {
    static const bool enabled = std::getenv("MOTO_TRACE_COMPOSE") != nullptr;
    return enabled;
}

bool same_derivative_role(const sym &from, const sym &to) {
    return in_field(from.field(), primal_fields) == in_field(to.field(), primal_fields);
}
} // namespace

struct generic_func::remap_cache {
    std::mutex mutex;
    std::map<remap_key, shared_expr> remap_by_key;
};

generic_func::generic_func() : remap_cache_(std::make_unique<remap_cache>()) {}

generic_func::generic_func(const std::string &name, approx_order order, size_t dim, field_t field)
    : expr(name, dim, field),
      order_(order),
      remap_cache_(std::make_unique<remap_cache>()) {}

generic_func::generic_func(const std::string &name, const var_inarg_list &in_args, const cs::SX &out,
                           approx_order order, field_t field)
    : generic_func(name, order, (size_t)out.size1(), field) {
    assert(out.size2() == 1 && "generic_constr output cols must be 1");
    set_from_casadi(in_args, out);
}

generic_func::generic_func(const generic_func &rhs)
    : expr(rhs),
      field_layout_store<var_list>(rhs),
      gen_(rhs.gen_),
      zero_dim_(rhs.zero_dim_),
      order_(rhs.order_),
      in_args_(rhs.in_args_),
      enable_if_all_deps_(rhs.enable_if_all_deps_),
      disable_if_any_deps_(rhs.disable_if_any_deps_),
      enable_if_any_deps_(rhs.enable_if_any_deps_),
      skip_unused_arg_check_(rhs.skip_unused_arg_check_),
      jac_sp_(rhs.jac_sp_),
      hess_sp_(rhs.hess_sp_),
      default_hess_sp_(rhs.default_hess_sp_),
      detect_jacobian_sparsity_(rhs.detect_jacobian_sparsity_),
      remap_cache_(std::make_unique<remap_cache>()),
      value(rhs.value),
      jacobian(rhs.jacobian),
      hessian(rhs.hessian) {}

generic_func::generic_func(generic_func &&) noexcept = default;
generic_func::~generic_func() = default;

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
    if (!value) {
        throw std::runtime_error(fmt::format("Function {} has no value implementation, please implement it or load from shared library", name()));
    }
    value(data);
}
void generic_func::jacobian_impl(func_approx_data &data) const {
    if (!jacobian) {
        throw std::runtime_error(fmt::format("Function {} has no jacobian implementation, please implement it or load from shared library", name()));
    }
    jacobian(data);
}

void generic_func::setup_hess() {
}

void generic_custom_func::finalize_impl() {
    if (in_field(field_, moto::func_fields) || field_ == __undefined) {
        throw std::runtime_error(fmt::format("func {} field type {} not qualified as custom function - finalization failed",
                                             name_, field::name(field_)));
    }
    generic_func::finalize_impl();
}

generic_custom_func::clone_ptr generic_custom_func::clone() const {
    return new generic_custom_func(*this);
}

void generic_func::hessian_impl(func_approx_data &data) const {
    if (!hessian) {
        throw std::runtime_error(fmt::format("Function {} has no hessian implementation, please implement it or load from shared library", name()));
    }
    hessian(data);
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

void generic_func::substitute_argument(const sym &arg, const sym &rhs) {
    field_write_guard();
    substitute(arg, rhs);
}

void generic_func::add_argument(const sym &in) {
    field_write_guard(in.field());
    if (std::find(in_args_.begin(), in_args_.end(), in) != in_args_.end()) {
        return;
    }
    auto &s = in_args_.emplace_back(in);
    add_dep(s);
}

void generic_func::add_argument(const var &in) {
    add_argument(static_cast<const sym &>(in));
}

void generic_func::add_arguments(const var_inarg_list &args) {
    for (sym &in : args) {
        add_argument(in);
    }
}

void generic_func::load_external(const std::string &path) {
    field_write_guard();
    load_external_impl(path);
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

generic_func::normalized_remap generic_func::normalize_argument_remap(const symbol_remap &remap) const {
    normalized_remap normalized;
    for (const auto &[from, to] : remap) {
        const auto from_uid = static_cast<size_t>(from->uid());
        const auto to_uid = static_cast<size_t>(to->uid());
        if (from_uid == to_uid) {
            continue;
        }
        if (!same_derivative_role(from, to)) {
            throw std::runtime_error(fmt::format(
                "func {} remap failed: cannot remap {} field {} to {} field {}; remap_arguments only supports primal-to-primal or nonprimal-to-nonprimal mappings",
                name_, from->name(), field::name(from->field()), to->name(), field::name(to->field())));
        }
        if (from->dim() != to->dim() || from->tdim() != to->tdim()) {
            throw std::runtime_error(fmt::format(
                "func {} remap failed: cannot remap {} dim {} tdim {} to {} dim {} tdim {}; remapped arguments must have matching dimensions",
                name_, from->name(), from->dim(), from->tdim(), to->name(), to->dim(), to->tdim()));
        }
        auto [it, inserted] = normalized.entries.emplace(from_uid, std::pair{from, to});
        if (!inserted && it->second.second->uid() != to_uid) {
            throw std::runtime_error(fmt::format(
                "func {} remap failed: source symbol uid {} maps to multiple targets",
                name_, from_uid));
        }
    }

    normalized.key.reserve(normalized.entries.size());
    for (const auto &[from_uid, entry] : normalized.entries) {
        normalized.key.emplace_back(from_uid, static_cast<size_t>(entry.second->uid()));
    }
    return normalized;
}

void generic_func::apply_argument_remap(const normalized_remap &remap,
                                        std::string_view context,
                                        size_t problem_uid) {
    for (const auto &[_, entry] : remap.entries) {
        const auto &[from, to] = entry;
        if (lowering_trace_enabled()) {
            if (problem_uid == static_cast<size_t>(-1)) {
                fmt::print("remapping {}: {} -> {}\n",
                           context, from->name(), to->name());
            } else {
                fmt::print("remapping {} in composed ocp uid {}: {} -> {}\n",
                           context, problem_uid, from->name(), to->name());
            }
        }
        substitute_argument(static_cast<const sym &>(from), static_cast<const sym &>(to));
    }
}

shared_expr generic_func::remap_arguments_cached(const symbol_remap &remap,
                                                 std::string_view context,
                                                 size_t problem_uid) {
    auto normalized = normalize_argument_remap(remap);
    if (normalized.empty())
        return shared_expr(*this);

    std::lock_guard lock(remap_cache_->mutex);
    auto &remaps = remap_cache_->remap_by_key;
    if (auto it = remaps.find(normalized.key); it != remaps.end()) {
        return it->second;
    }

    shared_expr remapped_expr(clone());
    auto &remapped_func = remapped_expr.as<generic_func>();
    remapped_func.apply_argument_remap(normalized, context, problem_uid);
    if (!remapped_func.finalize()) {
        throw std::runtime_error(fmt::format("func {} remap failed: remapped clone could not be finalized", name_));
    }

    auto [it, inserted] = remaps.emplace(std::move(normalized.key), remapped_expr);
    static_cast<void>(inserted);
    return it->second;
}

shared_expr generic_func::lower_expr_x_to_y_cached(std::string_view context, size_t problem_uid) {
    symbol_remap remap;
    remap.reserve(in_args_.size());
    for (const sym &arg : in_args_) {
        if (arg.field() == __x) {
            remap.emplace_back(var(arg), arg.next());
        }
    }
    return remap_arguments_cached(remap, context, problem_uid);
}

shared_expr generic_func::remap_arguments(const symbol_remap &remap) {
    return remap_arguments_cached(remap, "remap_arguments");
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

void generic_func::rebuild_argument_layout() {
    clear_entries();
    size_t idx = 0;
    for (auto &s : in_args_) {
        append_indexed_entry(s, idx++);
    }
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
    rebuild_argument_layout();
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
        t.jac_sp = nullptr;
        t.hess_sp = &hess_sp_;
        t.verbose = false;
        t.force_recompile = false;
        t.keep_generated_src = true;
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
const var_list &generic_func::in_args(field_t f) const { field_read_guard(f); return field_entries(f); }
size_t generic_func::arg_num(field_t f) const { field_read_guard(f); return field_entry_count(f); }
size_t generic_func::arg_dim(field_t f) const { field_read_guard(f); return field_dim(f); }
size_t generic_func::arg_tdim(field_t f) const { field_read_guard(f); return field_tdim(f); }
bool generic_func::has_arg(const sym &s) const {
    field_read_guard(s.field());
    return has_entry(s);
}
size_t generic_func::arg_idx(const sym &s) const {
    field_read_guard(s.field());
    return entry_index(s);
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
            goto CHECK_DONE;
        }
    }
    for (const auto &e : enable_if_all_deps_) {
        if (!prob->is_active(e)) {
            pass_check = false;
            goto CHECK_DONE;
        }
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
    if (rhs.task_) {
        task_ = new task_type(*rhs.task_);
    }
}

generic_func::gen_info &generic_func::gen_info::operator=(const gen_info &rhs) {
    if (this == &rhs) {
        return *this;
    }
    if (task_) {
        delete task_.get();
        task_ = nullptr;
    }
    if (rhs.task_) {
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
} // namespace moto
