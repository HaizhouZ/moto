#include <moto/core/external_function.hpp>
#include <moto/ocp/impl/func.hpp>
#include <moto/utils/codegen.hpp>

namespace moto {
static bool impl_func_gen_delegated_ = false; ///< true if the codegen is delegated to @ref moto::func_codegen

func_approx_map_ptr_t func_base::create_approx_map(sym_data &primal, dense_approx_data &raw, shared_data &shared) const {
    if (field() - __dyn >= field::num_func)
        throw std::runtime_error(fmt::format("create_approx_map cannot be called for func {} type {}",
                                             name(), field::name(field())));
    if (in_args().empty())
        throw std::runtime_error(fmt::format("in args unset for func {} in field {}",
                                             name(), field::name(field())));
    return std::make_unique<func_approx_map>(primal, raw, shared, *this);
}
void func_base::load_external_impl(const std::string &path) {
    auto funcs = load_approx(name_, true, order_ >= approx_order::first, order_ >= approx_order::second);
    value = [eval = funcs[0]](func_approx_map &d) {
        eval.invoke(d.in_arg_data(), d.v_);
    };
    jacobian = [jac = funcs[1]](func_approx_map &d) {
        jac.invoke(d.in_arg_data(), d.jac_);
    };

    hessian = [hess = funcs[2]](func_approx_map &d) {
        hess.invoke(d.in_arg_data(), d.hess_);
    };
}
void func_base::substitute(const sym &arg, const sym &rhs) {
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
void func_base::set_from_casadi(const var_inarg_list &in_args, const cs::SX &out) {
    add_arguments(in_args);
    gen_.out_ = out;
}
static utils::cs_codegen::worker_list func_codegen_workers_;
void func_base::finalize_impl() {
    if (!gen_.out_.is_empty()) {
        func_codegen::make_codegen_task(this);
        if (!impl_func_gen_delegated_) {
            func_codegen_workers_.wait_until_finished();
            load_external_impl();
        }
    }
}
void func_codegen::make_codegen_task(func_base *f) {
    utils::cs_codegen::task t;
    t.func_name = f->name_;
    t.sx_inputs = f->in_args_;
    t.sx_output = f->gen_.out_;
    t.gen_eval = f->order_ >= approx_order::zero;
    t.gen_jacobian = f->order_ >= approx_order::first;
    t.gen_hessian = f->order_ >= approx_order::second;
    t.append_value = f->field_ == __cost;
    t.append_jac = f->field_ == __cost;
    t.verbose = true;
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
    omp_set_num_threads(n_thread_bak);
}
} // namespace moto