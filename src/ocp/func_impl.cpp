#include <moto/core/external_function.hpp>
#include <moto/ocp/impl/func.hpp>
#include <moto/utils/codegen_bind.hpp>

namespace moto {
namespace impl {
sp_approx_map_ptr_t func::make_approx_map(sym_data &primal, approx_storage &raw, shared_data &shared) {
    if (field_ - __dyn >= field::num_func)
        throw std::runtime_error(fmt::format("make_approx_map cannot be called for func {} type {}",
                                             name_, field::name(field_)));
    if (in_args_.empty())
        throw std::runtime_error(fmt::format("in args unset for func {} in field {}",
                                             name_, field::name(field_)));
    return std::make_unique<sp_approx_map>(primal, raw, shared, *this);
}
void func::load_external(const std::string &path) {
    auto funcs = load_approx(name_, true, order() >= approx_order::first, order() >= approx_order::second);
    value = [eval = funcs[0]](sp_approx_map &d) {
        eval.invoke(d.in_arg_data(), d.v_);
    };
    jacobian = [jac = funcs[1]](sp_approx_map &d) {
        jac.invoke(d.in_arg_data(), d.jac_);
    };

    hessian = [hess = funcs[2]](sp_approx_map &d) {
        hess.invoke(d.in_arg_data(), d.hess_);
    };
}
void func::substitute(const sym &arg, const sym &rhs) {
    if (!gen_.out_.is_empty()) {
        gen_.out_ = cs::SX::substitute(gen_.out_, arg, rhs);
    }
    auto nh = sym_uid_idx_.extract(arg->uid_);
    nh.key() = rhs->uid_;
    sym_uid_idx_.insert(std::move(nh));         // update the uid index
    in_args_[sym_uid_idx_.at(rhs->uid_)] = rhs; // update the in_args_ to point to the new sym
}
void func::set_from_casadi(std::initializer_list<sym> in_args, const cs::SX &out) {
    add_arguments(in_args);
    gen_.out_ = out;
}
void func::finalize_impl() {
    if (!gen_.out_.is_empty()) {
        if (!gen_delegated_) {
            gen_.res_ = utils::generate_n_compile(name_, in_args_, {gen_.out_},
                                                  order_ >= approx_order::zero,
                                                  order_ >= approx_order::first,
                                                  order_ >= approx_order::second,
                                                  field_ == __cost); // generate code
            gen_.res_.wait();                                        // wait until codegen is done
            load_external();
        } else
            func_codegen::add(this);
    }
}
} // namespace impl
void func_codegen::enable() {
    impl::func::gen_delegated_ = true; // enable codegen delegation
}
void func_codegen::wait_until_all_compiled(size_t njobs) {
    std::vector<impl::func *> jobs;
    size_t cnt = 0;
    for (auto it_f = funcs_.begin(); it_f != funcs_.end(); ++it_f) {
        jobs.push_back(*it_f);
        cnt++;
        if (cnt == njobs || it_f + 1 == funcs_.end()) {
            for (auto f : jobs) {
                f->gen_.res_ = utils::generate_n_compile(f->name_, f->in_args_, {f->gen_.out_},
                                                         f->order_ >= approx_order::zero,
                                                         f->order_ >= approx_order::first,
                                                         f->order_ >= approx_order::second,
                                                         f->field_ == __cost);
            }
            for (auto f : jobs) {
                if (f->gen_.res_.valid()) {
                    f->gen_.res_.wait(); // wait until codegen is done
                    f->load_external();
                }
            }
            cnt = 0; // reset counter
            jobs.clear();
        }
    }

    funcs_.clear();
}
} // namespace moto