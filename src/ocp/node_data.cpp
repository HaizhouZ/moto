#include <moto/ocp/impl/custom_func.hpp>
#include <moto/ocp/impl/node_data.hpp>
#include <moto/ocp/problem.hpp>
#include <moto/ocp/soft_constr.hpp>
#include <moto/solver/data_base.hpp>
#define SHOW_DETAIL_TIMING
#include <moto/utils/timed_block.hpp>

namespace moto {
sym_data::sym_data(ocp *prob) : prob_(prob) {
    prob->wait_until_ready();
    auto set_default_val = [this](const sym &s) {
        if (s.default_value().size() > 0) {
            auto v = this->prob_->extract(this->value_.at(s.field()), s);
            if (s.default_value().size() != s.dim())
                throw std::runtime_error(fmt::format("default value size mismatch for sym {} in field {}, expected {}, got {}",
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
        return prob_->extract(value_.at(s.field()), s);
}

node_data::node_data(const ocp_ptr_t &prob)
    : prob_(prob),
      sym_(new sym_data(prob.get())),
      dense_(new lag_data(prob.get())),
      shared_(new shared_data(prob.get(), sym_.get())) {
    for (size_t field : func_fields) {
        size_t idx = 0;
        for (const generic_func &f : prob->exprs(field)) {
            sparse_[f.field()].push_back(f.create_approx_data(*sym_, *dense_, *shared_));
        }
    }
}
void node_data::update_approximation(update_mode config, bool include_original_cost) {
    /// @todo: always eval residual?
    // call to precompute
    bool update_cost = config == update_mode::eval_val || config == update_mode::eval_all;
    const bool reset_lag_jac = config != update_mode::eval_val && !include_original_cost;
    detail_timed_block_start("node_update_reset");
    if (update_cost) {
        dense_->cost_ = 0.;
        dense_->lag_ = 0.;
    }
    // set lagrangian gradient to zero
    if (config != update_mode::eval_val) {
        for (auto field : primal_fields) {
            if (reset_lag_jac)
                dense_->lag_jac_[field].setZero();
            dense_->lag_jac_corr_[field].setZero();
            dense_->cost_jac_[field].setZero();
        }

        if (config == update_mode::eval_hess ||
            config == update_mode::eval_derivatives ||
            config == update_mode::eval_all)
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
    detail_timed_block_end("node_update_reset");

    detail_timed_block_start("node_update_pre");
    for (const generic_custom_func &f : prob_->exprs(__pre_comp)) {
        f.custom_call((*shared_)[f]); ///< @todo pass update mode
    }
    detail_timed_block_end("node_update_pre");
    bool no_eval = config != update_mode::eval_val && config != update_mode::eval_all;
    bool no_jac = config != update_mode::eval_jac &&
                  config != update_mode::eval_derivatives &&
                  config != update_mode::eval_all;
    bool no_hess = config != update_mode::eval_hess &&
                   config != update_mode::eval_derivatives &&
                   config != update_mode::eval_all;
    const bool do_eval = !no_eval;
    const bool do_jac = !no_jac;
    const bool do_hess = !no_hess;
    detail_timed_block_start("node_update_funcs");
    for_each<std::array{__dyn, __eq_x, __eq_xu, __ineq_x, __ineq_xu, __eq_x_soft, __eq_xu_soft, __cost}>(
        [&](const generic_func &_f, func_approx_data &data) {
        _f.compute_approx(data,
                          do_eval && _f.order() >= approx_order::zero,
                          do_jac && _f.order() >= approx_order::first,
                          do_hess && _f.order() >= approx_order::second);
    });
    detail_timed_block_end("node_update_funcs");

    detail_timed_block_start("node_update_post");
    for (const generic_custom_func &f : prob_->exprs(__post_comp)) {
        f.custom_call((*shared_)[f]); ///< @todo pass update mode
    }
    detail_timed_block_end("node_update_post");
    if (config != update_mode::eval_val && include_original_cost)
        for (auto field : primal_fields)
            dense_->lag_jac_[field] = dense_->cost_jac_[field];

    detail_timed_block_start("node_update_merge");
    for (auto f : lag_data::stored_constr_fields) {
        if (prob_->dim(f) == 0)
            continue; // skip empty jacobian
        dense_->lag_ += dense_->approx_[f].v_.dot(dense_->dual_[f]);
        if (config >= update_mode::eval_jac)
            for (auto p : primal_fields) {
                if (dense_->approx_[f].jac_[p].is_empty())
                    continue; // skip empty jacobian
                dense_->approx_[f].jac_[p].right_T_times(dense_->dual_[f], dense_->lag_jac_[p]);
            }
    }
    detail_timed_block_end("node_update_merge");
    if (update_cost) {
        detail_timed_block_start("node_update_residuals");
        inf_prim_res_ = 0.;
        prim_res_l1_ = 0.;
        for (const auto &field_data : dense_->approx_) {
            if (field_data.v_.size() == 0)
                continue; // skip empty fields
            inf_prim_res_ = std::max(field_data.v_.cwiseAbs().maxCoeff(), inf_prim_res_);
            prim_res_l1_ += field_data.v_.lpNorm<1>();
        }
        inf_comp_res_ = 0.;
        for (const auto &comp : dense_->comp_) {
            if (comp.size() == 0)
                continue; // skip empty fields
            inf_comp_res_ = std::max(comp.cwiseAbs().maxCoeff(), inf_comp_res_);
        }
        dense_->lag_ += dense_->cost_;
        detail_timed_block_end("node_update_residuals");
    }
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
