#include <moto/core/external_function.hpp>
#include <moto/ocp/sym.hpp>
#include <moto/utils/codegen.hpp>

namespace moto {
sym::sym(const std::string &name, size_t dim, field_t type, default_val_t default_val)
    : expr(name, dim, type), cs::SX(cs::SX::sym(name, dim)) {
    if (!(size_t(type) <= field::num_sym || type == __usr_var))
        throw std::runtime_error(fmt::format("Invalid field {} for symbolic variable {}", type, name));
    set_default_value(default_val);
}
var sym::clone(const std::string &name) const {
    if (field_ != __x && field_ != __y)
        return var(sym(*this));
    else { // clone also dual
        return clone_states<sym>(name);
    }
} ///< clone the symbolic variable
void sym::integrate(vector_ref x, vector_ref dx, vector_ref out, scalar_t alpha) const {
    if (!integrator_)
        out.noalias() = x + alpha * dx;
    else
        (*integrator_)(x, dx, out, alpha);
} ///< integrate the variable by dx with step size alpha

void sym::difference(vector_ref x1, vector_ref x0, vector_ref out) const {
    if (!differencer_)
        out.noalias() = x1 - x0;
    else
        (*differencer_)(x1, x0, out);
} ///< compute the difference between two variables x1 - x0
void sym::set_default_value(const default_val_t &default_val) {
    if (std::holds_alternative<vector>(default_val)) {
        auto &v = std::get<vector>(default_val);
        if (v.size() != dim())
            throw std::runtime_error(fmt::format("default value size mismatch for sym {} in field {}, expected {}, got {}",
                                                 name(), field::name(field()), dim(), v.size()));
        default_value_ = std::move(v);
    } else if (std::holds_alternative<scalar_t>(default_val)) {
        default_value_ = vector::Constant(dim(), std::get<scalar_t>(default_val));
    } /// leave empty
}
void sym::finalize_impl() {
    if (field_ == __x && !bool(dual_)) {
        throw std::runtime_error("dual pointer should not be null when field == __x");
    } else if (field_ == __y && !bool(dual_)) {
        throw std::runtime_error("dual pointer should be null when field == __y");
    }
    bool is_state = (field_ == __x || field_ == __y);
    bool wait_codegen = is_state; // wait for codegen only for state variables
    if (field_ == __x) {
        utils::cs_codegen::job_list workers;
        if (!integrator_) { // maybe set by derived class
            auto dx = sym::usr_var(name_ + "_dx", tdim_);
            auto step = sym::usr_var(name_ + "_stepsize", 1);
            auto out = symbolic_integrate(*this, dx * step);                                    // dimension is dim_
            if (tdim_ != dim_ ||                                                                // if not equal, should not call default integrate
                !cs::SX::simplify(out - sym::symbolic_integrate(*this, dx * step)).is_zero()) { // if not default, generate function
                has_non_trivial_integration_ = true;
                dual_->has_non_trivial_integration_ = true;
                utils::cs_codegen::task int_gen_task;
                int_gen_task.func_name = name_ + "_integrate";
                int_gen_task.sx_inputs = {*this, dx, step};
                int_gen_task.sx_output = out;
                workers.add(std::move(utils::cs_codegen::generate_and_compile(int_gen_task)
                                          .add_callback([this, func_name = int_gen_task.func_name]() {
                                              ext_func f(func_name);
                                              integrator_.reset(new std::function(
                                                  [f = std::move(f)](vector_ref x, vector_ref dx, vector_ref out, scalar_t alpha) {
                                                      std::vector<vector_ref> inputs = {x, dx, vector_ref(mapped_vector(&alpha, 1))};
                                                      f.invoke(inputs, out);
                                                  }));
                                          })));
            }
        }
        if (!differencer_) { // maybe set by derived class
            auto x0 = sym::usr_var(name_ + "_arg0", dim_);
            auto x1 = sym::usr_var(name_ + "_arg1", dim_);
            auto out = symbolic_difference(x1, x0);
            if (out.size1() != tdim_)
                throw std::runtime_error(fmt::format("difference function output dimension mismatch for sym {} in field {}, expected {}, got {}",
                                                     name(), field::name(field_), tdim_, out.size1()));
            if (tdim_ != dim_ || !cs::SX::simplify(out - sym::symbolic_difference(x1, x0)).is_zero()) { // if not default, generate function
                has_non_trivial_difference_ = true;
                dual_->has_non_trivial_difference_ = true;
                utils::cs_codegen::task diff_gen_task;
                diff_gen_task.func_name = name_ + "_difference";
                diff_gen_task.sx_inputs = {x1, x0};
                diff_gen_task.sx_output = out;
                workers.add(std::move(utils::cs_codegen::generate_and_compile(diff_gen_task)
                                          .add_callback([this, func_name = diff_gen_task.func_name]() {
                                              ext_func f(func_name);
                                              differencer_.reset(new std::function(
                                                  [f = std::move(f)](vector_ref x1, vector_ref x0, vector_ref out) {
                                                      std::vector<vector_ref> inputs = {x1, x0};
                                                      f.invoke(inputs, out);
                                                  }));
                                          })));
            }
        }
        if (workers.jobs.empty()) {
            set_ready_status(true);
        } else {
            utils::cs_codegen::server::add_job(std::move(
                workers.add_finish_callback([this]() {
                    set_ready_status(true);
                })));
        }
    } else if (field_ == __y) {
        utils::cs_codegen::server::add_job(
            [this]() {
                if (!dual_->wait_until_ready()) {
                    throw std::runtime_error(fmt::format("dual sym {} in field {} with uid {} is not ready",
                                                         dual_->name(), field::name(dual_->field()), dual_->uid()));
                }
                integrator_ = dual_->integrator_;
                differencer_ = dual_->differencer_;
                set_ready_status(true);
            });
    } else
        set_ready_status(true);
}
} // namespace moto