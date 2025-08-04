#include <fmt/ranges.h>
#include <moto/core/external_function.hpp>
#include <moto/ocp/impl/func.hpp>
#include <moto/utils/codegen.hpp>
namespace moto {
static bool impl_func_gen_delegated_ = false; ///< true if the codegen is delegated to @ref moto::func_codegen

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
    if (!gen_.out_.is_empty()) {
        gen_.out_ = cs::SX::substitute(gen_.out_, arg, rhs);
    }
    auto nh = sym_uid_idx_.extract(arg.uid());
    nh.key() = rhs.uid();
    auto it = sym_uid_idx_.insert(std::move(nh)); // update the uid index
    assert(it.inserted && "substitute failed");
    in_args_.at(it.position->second) = rhs; // update the in_args_ to point to the new sym
    dep_.at(it.position->second) = rhs;     // update the dep_ to point to the new sym
}
void generic_func::set_from_casadi(const var_inarg_list &in_args, const cs::SX &out) {
    add_arguments(in_args);
    gen_.out_ = out;
}
static utils::cs_codegen::job_list func_codegen_workers_;
std::vector<generic_func *> cg_funcs_;
void generic_func::finalize_impl() {
    if (order_ != approx_order::none && dim_ == dim_tbd) {
        throw std::runtime_error(fmt::format("generic_func {} has no dimension set", name_));
    }
    if (!gen_.out_.is_empty()) {
        func_codegen::make_codegen_task(this);
        if (!impl_func_gen_delegated_) {
            func_codegen_workers_.wait_until_finished();
            load_external_impl();
        } else
            cg_funcs_.push_back(this); // will be loaded later
    } else {
        bool value_missing = order_ >= approx_order::zero && !value;
        bool jacobian_missing = order_ >= approx_order::first && !jacobian;
        bool hessian_missing = order_ >= approx_order::second && !hessian;
        if (value_missing || jacobian_missing || hessian_missing) {
            std::vector<std::string> missing;
            missing.reserve(3);
            if (value_missing)
                missing.push_back("value");
            if (jacobian_missing)
                missing.push_back("jacobian");
            if (hessian_missing)
                missing.push_back("hessian");
            throw std::runtime_error(fmt::format("generic_func {} has no codegen set for [{}]", name_, fmt::join(missing, ",")));
        }
    }
}
void func_codegen::make_codegen_task(generic_func *f) {
    utils::cs_codegen::task t;
    t.func_name = f->name_;
    t.sx_inputs = f->in_args_;
    t.sx_output = f->gen_.out_;
    t.gen_eval = f->order_ >= approx_order::zero;
    t.gen_jacobian = f->order_ >= approx_order::first;
    t.gen_hessian = f->order_ >= approx_order::second;
    t.append_value = f->field_ == __cost;
    t.append_jac = f->field_ == __cost;
    t.verbose = false;
    t.force_recompile = false;
    t.keep_generated_src = false;
    auto workers = utils::cs_codegen::generate_and_compile(std::move(t));
    if (impl_func_gen_delegated_) {
        func_codegen_workers_.add(std::move(workers));
    } else {
        func_codegen_workers_.add(workers);
    }
}

void func_codegen::enable() {
    impl_func_gen_delegated_ = true; // enable codegen delegation
}

void func_codegen::wait_until_all_compiled(size_t njobs) {
    size_t n_thread_bak = omp_get_num_threads();
    omp_set_num_threads(njobs);
    func_codegen_workers_.wait_until_finished();
    for (auto f : cg_funcs_) {
        f->load_external_impl();
    }
    cg_funcs_.clear();
    omp_set_num_threads(n_thread_bak);
}
} // namespace moto