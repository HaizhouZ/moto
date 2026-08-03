#include <moto/ocp/impl/custom_func.hpp>
#include <moto/ocp/impl/node_data.hpp>
#include <moto/ocp/problem.hpp>
#include <moto/ocp/soft_constr.hpp>
#include <moto/solver/data_base.hpp>
#include <moto/core/linear_backend.hpp>

namespace moto {
struct node_linear_plan {
  struct condensation_phase {
    linear_backend::batch_condensation_kernel kernel;
    std::vector<scalar_t *> pointers;
    std::vector<vector> zeros;
    bool built = false;
  };
  linear_backend::batch_product_kernel constraint_gradient;
  std::vector<scalar_t *> pointers;
  bool gradient_built = false;
  std::array<condensation_phase, 2> condensation;
  linear_backend::batch_product_kernel jacobian_steps;
  std::vector<scalar_t *> jacobian_step_pointers;
  bool jacobian_steps_built = false;
};
sym_data::sym_data(ocp *prob) : prob_(prob) {
    prob->wait_until_ready();
    auto set_default_val = [this](const sym &s) {
        if (s.default_value().size() > 0) {
            auto v = this->prob_->extract(this->value_[s.field()], s);
            if (s.default_value().size() != s.dim())
                throw std::runtime_error(
            fmt::format("default value size mismatch for sym {} in field {}, "
                        "expected {}, got {}",
                                                     s.name(), field::name(s.field()), s.dim(), s.default_value().size()));
            v = s.default_value();
        }
    };
    for (size_t i = 0; i < field::num_sym; i++) {
        value_[i].resize(prob_->dim(i));
        value_[i].setZero();
        for (const sym &s : prob_->exprs(static_cast<field_t>(i))) {
            set_default_val(s);
        }
    }
    for (const sym &s : prob_->exprs(__usr_var)) {
        set_default_val(s);
    }
}

void sym_data::integrate(field_t f, vector &dx, scalar_t alpha) {
    assert(dx.size() == prob_->tdim(f) && "dx size mismatch");
    for (const sym &s : prob_->exprs(f)) {
        auto v = get(s);
        s.integrate(v, prob_->extract_tangent(dx, s), v, alpha);
    }
}

void sym_data::print() {
    auto p = prob_;
    for (auto f : concat_fields(primal_fields, std::array{__s, __p, __usr_var})) {
        if (p->dim(f) == 0)
            continue; // skip empty fields
        fmt::println("Field {}: dim {}", field::name(f), p->dim(f));
        for (const sym &s : p->exprs(f)) {
            fmt::println("{}: dim {} value {}", s.name(), s.dim(), get(s).transpose());
        }
    }
}
vector_ref sym_data::get(const sym &s) {
    if (s.field() == __usr_var)
        return usr_value_.at(s.uid());
    else
        return prob_->extract(value_[s.field()], s);
}

node_data::node_data(const ocp_ptr_t &prob)
    : prob_(prob),
      sym_(new sym_data(prob.get())),
      dense_(new lag_data(prob.get())),
      shared_(new shared_data(prob.get(), sym_.get())) {
    const auto &profile = prob->linear_profile();
    for (const auto cf : constr_fields)
        for (const auto pf : primal_fields)
            dense_->approx_[cf].jac_[pf].plan(
                profile.get(linear_target::jacobian, cf, pf));
    for (const auto a : primal_fields) for (const auto b : primal_fields) {
        dense_->lag_hess_[a][b].plan(
            profile.get(linear_target::lag_hessian, a, b));
        dense_->hessian_modification_[a][b].plan(
            profile.get(linear_target::hessian_modification, a, b));
    }
    for (size_t field : func_fields) {
        for (const generic_func &f : prob->exprs(field)) {
      sparse_[f.field()].push_back(
          f.create_approx_data(*sym_, *dense_, *shared_));
        }
    }
}
void node_data::update_approximation(update_mode config, bool include_original_cost) {
    /// @todo: always eval residual?
    // call to precompute
    const bool eval_value = config == update_mode::eval_val || config == update_mode::eval_all;
    const bool eval_jacobian = config == update_mode::eval_jac ||
                               config == update_mode::eval_derivatives ||
                               config == update_mode::eval_all;
    const bool eval_hessian = config == update_mode::eval_hess ||
                              config == update_mode::eval_derivatives ||
                              config == update_mode::eval_all;
    const bool eval_derivatives = eval_jacobian || eval_hessian;
    const bool reset_lag_jac = eval_derivatives && !include_original_cost;
    if (eval_value) {
        dense_->cost_ = 0.;
        dense_->lag_ = 0.;
    }
    // set lagrangian gradient to zero
    if (eval_derivatives) {
        for (auto field : primal_fields) {
            if (reset_lag_jac)
                dense_->lag_jac_[field].setZero();
            dense_->lag_jac_corr_[field].setZero();
            dense_->cost_jac_[field].setZero();
        }

        if (eval_hessian) {
            for (auto &hess_l_0 : dense_->lag_hess_) {
                for (auto &hess_l_1 : hess_l_0) {
                    hess_l_1.setZero();
                }
            }
            for (auto &hess_l_0 : dense_->hessian_modification_) {
                for (auto &hess_l_1 : hess_l_0) {
                    hess_l_1.setZero();
                }
            }
        }
    }
    for (const generic_custom_func &f : prob_->exprs(__pre_comp)) {
        f.custom_call((*shared_)[f]); ///< @todo pass update mode
    }
  for_each<func_fields>(
      [=, this](const generic_func &_f, func_approx_data &data) {
        _f.compute_approx(data,
                          eval_value && _f.order() >= approx_order::zero,
                          eval_jacobian && _f.order() >= approx_order::first,
                          eval_hessian && _f.order() >= approx_order::second);
    });
  if (eval_jacobian)
    condense_soft_constraints(eval_hessian);
    for (const generic_custom_func &f : prob_->exprs(__post_comp)) {
        f.custom_call((*shared_)[f]); ///< @todo pass update mode
    }
    if (eval_derivatives && include_original_cost)
        for (auto field : primal_fields)
            dense_->lag_jac_[field] = dense_->cost_jac_[field];

    for (auto f : lag_data::stored_constr_fields) {
        if (prob_->dim(f) == 0)
            continue; // skip empty jacobian
        if (eval_value)
            dense_->lag_ += dense_->approx_[f].v_.dot(dense_->dual_[f]);
  }
        if (eval_jacobian)
    assemble_constraint_gradient();
    if (eval_value) {
        inf_prim_res_ = 0.;
        prim_res_l1_ = 0.;
        for (auto field : constr_fields) {
            size_t idx = 0;
            for (const generic_constr &c : prob_->exprs(field)) {
                const auto &cd = *sparse_[field][idx];
                const auto summary = c.primal_residual_summary(cd);
                inf_prim_res_ = std::max(inf_prim_res_, summary.inf);
                prim_res_l1_ += summary.l1;
                ++idx;
            }
        }
        inf_comp_res_ = 0.;
        for (const auto &comp : dense_->comp_) {
            if (comp.size() == 0)
                continue; // skip empty fields
            inf_comp_res_ = std::max(comp.cwiseAbs().maxCoeff(), inf_comp_res_);
        }
        dense_->lag_ += dense_->cost_;
    }
}

void node_data::prepare_soft_condensation(bool hessian) {
  if (!linear_plan_)
    linear_plan_ = std::make_shared<node_linear_plan>();
  auto &phase = linear_plan_->condensation[hessian];
  if (!phase.built) {
    linear_backend::batch_condensation_spec batch;
    for_each<ineq_soft_constr_fields>(
        [&](const soft_constr &sf, soft_constr::data_map_t &sd) {
          auto view = sf.condensation(sd, hessian);
          if (view.residuals.empty())
            return;
          linear_backend::condensation_spec spec;
          spec.rows = sf.dim();
          spec.residual_signs = std::move(view.residual_signs);
          std::vector<size_t> args;
          std::vector<size_t> remap(sf.in_args().size(), size_t(-1));
          for (size_t i = 0; i < sf.in_args().size(); ++i) {
            if (!sd.has_jacobian_block(i) || sd.lag_jac_corr_[i].size() == 0)
              continue;
            const auto &sp = sf.jac_sparsity()[i];
            remap[i] = args.size();
            args.push_back(i);
            spec.jacobians.push_back(
                {sp.pattern, sp.row_offset, sp.col_offset, sp.rows, sp.cols});
            phase.pointers.push_back(sd.jac_[i].data());
          }
          for (const auto *residual : view.residuals)
            phase.pointers.push_back(const_cast<scalar_t *>(residual));
          for (const auto *weight : view.weights) {
            if (weight) {
              phase.pointers.push_back(const_cast<scalar_t *>(weight));
            } else {
              phase.zeros.push_back(vector::Zero(sf.dim()));
              phase.pointers.push_back(phase.zeros.back().data());
            }
          }
          for (size_t i : args)
            phase.pointers.push_back(sd.lag_jac_corr_[i].data());
          for (size_t i : args) {
            for (size_t j : args) {
              if (sd.lag_hess_[i][j].size() == 0)
                continue;
              spec.hessian_pairs.emplace_back(remap[i], remap[j]);
              phase.pointers.push_back(sd.lag_hess_[i][j].data());
            }
          }
          batch.constraints.push_back(std::move(spec));
        });
    if (!batch.constraints.empty())
      phase.kernel = linear_backend::compile_batch_condensation(std::move(batch));
    phase.built = true;
  }
}

void node_data::condense_soft_constraints(bool hessian) {
  prepare_soft_condensation(hessian);
  auto &phase = linear_plan_->condensation[hessian];
  if (!phase.pointers.empty())
    phase.kernel(phase.pointers);
}

void node_data::prepare_constraint_gradient() {
  if (!linear_plan_)
    linear_plan_ = std::make_shared<node_linear_plan>();
  if (!linear_plan_->gradient_built) {
    auto &plan = *linear_plan_;
    linear_backend::batch_product_spec batch;
    for (auto f : lag_data::stored_constr_fields) {
      if (prob_->dim(f) == 0)
        continue;
      for (auto p : primal_fields) {
        const auto &jac = dense_->approx_[f].jac_[p];
        if (jac.is_empty())
          continue;
        auto pointers = linear_backend::panel_pointers(jac);
        plan.pointers.insert(plan.pointers.end(), pointers.begin(),
                             pointers.end());
        plan.pointers.push_back(dense_->dual_[f].data());
        plan.pointers.push_back(dense_->lag_jac_[p].data());
        batch.products.push_back({.sparse = linear_backend::describe(jac),
                                  .op = linear_backend::product_op::transpose_times,
                                  .other_rows = jac.rows(),
                                  .other_cols = 1,
                                  .out_rows = jac.cols(),
                                  .out_cols = 1});
      }
    }
    if (!batch.products.empty())
      plan.constraint_gradient =
          linear_backend::compile_batch_product(std::move(batch));
    plan.gradient_built = true;
  }
}

void node_data::assemble_constraint_gradient() {
  prepare_constraint_gradient();
  if (!linear_plan_->pointers.empty())
    linear_plan_->constraint_gradient(linear_plan_->pointers);
}

void node_data::prepare_soft_jacobian_steps() {
  if (!linear_plan_)
    linear_plan_ = std::make_shared<node_linear_plan>();
  auto &plan = *linear_plan_;
  if (!plan.jacobian_steps_built) {
    std::vector<std::pair<size_t, std::vector<linear_backend::panel_layout>>>
        products;
    for_each<ineq_soft_constr_fields>(
        [&](const soft_constr &sf, soft_constr::data_map_t &sd) {
          auto output = sf.jacobian_step(sd);
          if (output.size() == 0)
            return;
          std::vector<linear_backend::panel_layout> layouts;
          std::vector<size_t> args;
          for (size_t i = 0; i < sf.in_args().size(); ++i) {
            if (!sd.has_jacobian_block(i) || sd.prim_step_[i].size() == 0)
              continue;
            const auto &sp = sf.jac_sparsity()[i];
            layouts.push_back(
                {sp.pattern, sp.row_offset, sp.col_offset, sp.rows, sp.cols});
            args.push_back(i);
            plan.jacobian_step_pointers.push_back(sd.jac_[i].data());
          }
          for (size_t i : args)
            plan.jacobian_step_pointers.push_back(sd.prim_step_[i].data());
          plan.jacobian_step_pointers.push_back(output.data());
          products.emplace_back(sf.dim(), std::move(layouts));
        });
    if (!products.empty())
      plan.jacobian_steps =
          linear_backend::compile_batch_jacobian_product(std::move(products));
    plan.jacobian_steps_built = true;
  }
}

void node_data::evaluate_soft_jacobian_steps() {
  prepare_soft_jacobian_steps();
  if (!linear_plan_->jacobian_step_pointers.empty())
    linear_plan_->jacobian_steps(linear_plan_->jacobian_step_pointers);
}

void node_data::prepare_linear_plan() {
  prepare_constraint_gradient();
  prepare_soft_condensation(false);
  prepare_soft_condensation(true);
  prepare_soft_jacobian_steps();
}

void node_data::configure_scaling_profile(bool enabled) {
  bool changed = false;
  for (const auto cf : hard_constr_fields_non_dyn)
    for (const auto pf : primal_fields)
      changed |= dense_->approx_[cf].jac_[pf].set_dynamic_eye(enabled);
  if (changed)
    linear_plan_.reset();
}

void node_data::print_residuals() const {
    for (auto f : lag_data::stored_constr_fields) {
        fmt::println("Field {}: dim {} residual {}", field::name(f), dense_->approx_[f].v_.size(),
                     dense_->approx_[f].v_.transpose());
    }
}

void node_data::bind_soft_runtime_owner(solver::data_base *owner) {
    for (auto field : ineq_soft_constr_fields) {
        for (auto &ptr : sparse_[field]) {
            if (auto *sd = dynamic_cast<soft_constr::data_map_t *>(ptr.get())) {
                sd->solver_data_ = owner;
            }
        }
    }
}
} // namespace moto
