#include <moto/core/external_function.hpp>
#include <moto/ocp/impl/func.hpp>
#include <moto/ocp/impl/func_data.hpp>
#include <moto/ocp/problem.hpp>
#include <moto/utils/codegen.hpp>

namespace moto {
sym::sym(const std::string &name, size_t dim, field_t type, default_val_t default_val)
    : expr(name, dim, type), cs::SX(cs::SX::sym(name, dim)) {
    if (!(size_t(type) <= field::num_sym || type == __usr_var))
        throw std::runtime_error(fmt::format("Invalid field {} for symbolic variable {}", type, name));
    set_default_value(default_val);
}
var sym::clone() const {
    if (field_ != __x && field_ != __y)
        return var(sym(*this));
    else { // clone also dual
        return clone_states<sym>();
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
            auto dx = sym::usr_var(name_ + "dx", tdim_);
            auto step = sym::usr_var(name_ + "stepsize", 1);
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
                                                      std::vector<vector_ref> outputs = {out};
                                                      f.invoke(inputs, outputs);
                                                  }));
                                          })));
            }
        }
        if (!differencer_) { // maybe set by derived class
            auto x0 = sym::usr_var(name_ + "0", dim_);
            auto x1 = sym::usr_var(name_ + "1", dim_);
            auto out = symbolic_difference(x1, x0);
            if (out.size1() != tdim_)
                throw std::runtime_error(fmt::format("difference function output dimension mismatch for sym {} in field {}, expected {}, got {}",
                                                     name(), field::name(field_), tdim_, out.size1()));
            if (!cs::SX::simplify(out - sym::symbolic_difference(x1, x0)).is_zero()) { // if not default, generate function
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
                                                      std::vector<vector_ref> outputs = {out};
                                                      f.invoke(inputs, outputs);
                                                  }));
                                          })));
            }
        }
        if (workers.jobs.empty()) {
            set_ready_status(true);
        } else {
            utils::cs_codegen::server::add_job(std::move(
                workers.add_finish_callback([this]() { set_ready_status(true); })));
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
func_arg_map::func_arg_map(sym_data &primal, shared_data &shared, const generic_func &f)
    : func_(f), shared_(shared), sym_uid_idx_(f.sym_uid_idx_), primal_(&primal) {
    auto &in_args = f.in_args();
    in_args_.reserve(in_args.size());
    for (auto &arg : in_args) {
        static vector empty;
        if (problem()->is_active(arg))
            in_args_.push_back(primal[arg]);
        else
            in_args_.push_back(empty);
    }
}

vector_ref get_value_ref(const generic_func &f, merit_data &raw) {
    if (f.field() == __cost) {
        return vector_ref(mapped_vector(&raw.cost_, 1));
    } else if (in_field(f.field(), merit_data::stored_constr_fields)) {
        return raw.approx_[f.field()].v_.segment(raw.prob_->get_expr_start(f), f.dim());
    } else {
        throw std::runtime_error(fmt::format("Function {} in field {} with uid {} does not have stored value",
                                             f.name(), f.field(), f.uid()));
    }
}

func_approx_data::func_approx_data(sym_data &primal,
                                   merit_data &raw,
                                   shared_data &shared,
                                   const generic_func &f)
    : func_arg_map(primal, shared, f), v_(get_value_ref(f, raw)), merit_data_(&raw) {
    auto &in_args = f.in_args();
    size_t f_st = raw.prob_->get_expr_start(f);
    // for non-cost
    if (f.order() >= approx_order::first) {
        // bind merit jacobian
        merit_jac_.reserve(in_args_.size());
        for (size_t i : range(in_args_.size())) {
            if (in_args[i]->field() < field::num_prim && raw.prob_->is_active(in_args[i])) {
                merit_jac_.push_back(raw.prob_->extract_tangent(raw.jac_[in_args[i]->field()], in_args[i]));
            } else { // useless
                static row_vector empty;
                merit_jac_.push_back(empty);
            }
        }
        // bind approx jacobian
        if (f.field() != __cost) {
            static matrix empty;
            jac_.assign(in_args_.size(), empty);
        } else {
            jac_.reserve(merit_jac_.size());
            jac_.assign(merit_jac_.begin(), merit_jac_.end());
        }
    }
    setup_hessian();
}

void func_approx_data::setup_hessian() {
    auto &f = func_;
    auto &in_args = f.in_args();
    assert(merit_data_ != nullptr && "merit_data_ should not be null");
    auto &raw = *merit_data_;
    if (f.order() >= approx_order::second || in_field(f.field(), ineq_soft_constr_fields)) {
        size_t field_1, field_2;
        auto *hessian = f.field() == __cost ? &raw.hessian_ : &raw.hessian_modification_;
        merit_hess_.resize(in_args_.size());
        for (size_t i : range(in_args_.size())) {
            if (in_args[i]->field() < field::num_prim) {
                merit_hess_[i].reserve(in_args_.size());
                for (size_t j : range(in_args_.size())) {
                    field_1 = in_args[i]->field();
                    field_2 = in_args[j]->field();
                    if (raw.prob_->is_active(in_args[i]) &&
                        field_2 < field::num_prim &&
                        raw.prob_->is_active(in_args[j])) {
                        /// @note order matches merit_data
                        /// h[i][j] = h[j][i] if i, j in the same field or field(i) < field(j)
                        /// otherwise only keep h[i][j] (empty)
                        if (func_.hess_sp_[i][j] == sparsity::unknown) {
                            goto BIND_EMPTY_HESS;
                        } else if (field_1 >= field_2) {
                            merit_hess_[i].push_back((*hessian)[field_1][field_2].insert(
                                raw.prob_->get_expr_start_tangent(in_args[i]),
                                raw.prob_->get_expr_start_tangent(in_args[j]),
                                in_args[i]->tdim(), in_args[j]->tdim(), func_.hess_sp_[i][j]));
                            continue;
                        }
                    }
                BIND_EMPTY_HESS:
                    // this should be empty. do this anyway to make the shape of merit_hess_ right
                    static matrix empty;
                    merit_hess_[i].push_back(empty);
                }
            }
        }
        merit_hess_.shrink_to_fit();
    }
}
} // namespace moto