#include <moto/core/external_function.hpp>
#include <moto/ocp/impl/func.hpp>
#include <moto/utils/codegen.hpp>

namespace moto {
static bool impl_func_gen_delegated_ = false; ///< true if the codegen is delegated to @ref moto::func_codegen

func_approx_map_ptr_t func::create_approx_map(sym_data &primal, dense_approx_data &raw, shared_data &shared) const {
    if (field() - __dyn >= field::num_func)
        throw std::runtime_error(fmt::format("create_approx_map cannot be called for func {} type {}",
                                             name(), field::name(field())));
    if (in_args().empty())
        throw std::runtime_error(fmt::format("in args unset for func {} in field {}",
                                             name(), field::name(field())));
    return std::make_unique<func_approx_map>(primal, raw, shared, *this);
}
void func::load_external_impl(const std::string &path) {
    auto funcs = load_approx(name(), true, order() >= approx_order::first, order() >= approx_order::second);
    value() = [eval = funcs[0]](func_approx_map &d) {
        eval.invoke(d.in_arg_data(), d.v_);
    };
    jacobian() = [jac = funcs[1]](func_approx_map &d) {
        jac.invoke(d.in_arg_data(), d.jac_);
    };

    hessian() = [hess = funcs[2]](func_approx_map &d) {
        hess.invoke(d.in_arg_data(), d.hess_);
    };
}
void func::impl::substitute(const sym &arg, const sym &rhs) {
    if (!gen_.out_.is_empty()) {
        gen_.out_ = cs::SX::substitute(gen_.out_, arg, rhs);
    }
    auto nh = sym_uid_idx_.extract(arg.uid());
    nh.key() = rhs.uid();
    sym_uid_idx_.insert(std::move(nh));            // update the uid index
    in_args_.at(sym_uid_idx_.at(rhs.uid())) = rhs; // update the in_args_ to point to the new sym
}
void func::set_from_casadi(std::initializer_list<sym> in_args, const cs::SX &out) {
    add_arguments(in_args);
    shared().gen_.out_ = out;
}
void func::finalize_impl() {
    if (!shared().gen_.out_.is_empty()) {
        if (!impl_func_gen_delegated_) {
            shared().gen_.res_ = func_codegen::make_codegen_task(this);
            shared().gen_.res_.wait(); // wait until codegen is done
            load_external_impl();
        } else
            func_codegen::add(this);
    }
}
std::future<void> func_codegen::make_codegen_task(func *_f) {
    auto f = &_f->shared();
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
    t.force_recompile = true;
    auto workers = utils::cs_codegen::generate_and_compile(std::move(t));
    return std::async(std::launch::async, [w = std::move(workers)]() mutable {
        w.wait_until_finished();
    });
}

void func_codegen::add(func *f) { code_gen_funcs_.push_back(f); }

std::vector<func *> func_codegen::code_gen_funcs_ = {};

void func_codegen::enable() {
    impl_func_gen_delegated_ = true; // enable codegen delegation
}

void func_codegen::wait_until_all_compiled(size_t njobs) {
    std::vector<func *> jobs;
    size_t cnt = 0;
    auto &funcs_ = code_gen_funcs_;
    for (auto it_f = funcs_.begin(); it_f != funcs_.end(); ++it_f) {
        jobs.push_back(*it_f);
        cnt++;
        if (cnt == njobs || it_f + 1 == funcs_.end()) {
            for (auto f : jobs) {
                f->shared().gen_.res_ = make_codegen_task(f); // make codegen task for each function
            }
            for (auto f : jobs) {
                f->shared().gen_.res_.wait(); // wait until codegen is done
                f->load_external();
            }
            cnt = 0; // reset counter
            jobs.clear();
        }
    }

    funcs_.clear();
}
} // namespace moto