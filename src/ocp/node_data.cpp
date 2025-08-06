#include <moto/ocp/impl/custom_func.hpp>
#include <moto/ocp/impl/data_mgr.hpp>
namespace moto {
sym_data::sym_data(ocp *prob) : prob_(prob) {
    prob->wait_until_ready();
    for (size_t i = 0; i < field::num_sym; i++) {
        value_[i].resize(prob_->dim(i));
        value_[i].setZero();
        auto &syms = prob_->exprs(static_cast<field_t>(i));
        for (const sym &s : syms) {
            auto v = prob_->extract(value_[i], s);
            if (s.default_value().size() > 0) {
                assert(s.default_value().size() == s.dim() && "default value size mismatch");
                v = s.default_value();
            }
        }
    }
    for (const sym &s : prob_->exprs(__usr_var)) {
        auto &v = usr_value_[s.uid()] = vector::Zero(s.dim());
        if (s.default_value().size() > 0) {
            assert(s.default_value().size() == s.dim() && "default value size mismatch for usr_var");
            v = s.default_value();
        }
    }
}

void sym_data::print() {
    auto p = prob_;
    for (auto f : concat_fields(primal_fields, std::array{__p, __usr_var})) {
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
      dense_(new merit_data(prob.get())),
      shared_(new shared_data(prob.get(), sym_.get())) {
    for (size_t field : func_fields) {
        size_t idx = 0;
        for (const generic_func &f : prob->exprs(field)) {
            sparse_[f.field()].push_back(f.create_approx_data(*sym_, *dense_, *shared_));
        }
    }
}
void node_data::update_approximation(update_mode config) {
    /// @todo: always eval residual?
    // call to precompute
    bool update_cost = config == update_mode::eval_val || config == update_mode::eval_all;
    if (update_cost) {
        dense_->cost_ = 0.;
        dense_->merit_ = 0.;
    }
    // set merit jacobian to zero
    if (config > update_mode::eval_val) {
        for (auto &r : dense_->res_stat_) {
            r.setZero();
        }
        for (auto field : primal_fields) {
            dense_->jac_[field].setZero();
            dense_->jac_modification_[field].setZero();
        }
        for (auto &a : dense_->active_ineqs_) {
            a.clear();
        }

        for (auto &a : dense_->active_ineq_approx_) {
            a.dim = 0;
        }

        if (config >= update_mode::eval_hess)
            for (auto &hess_l_0 : dense_->hessian_) {
                for (auto &hess_l_1 : hess_l_0) {
                    hess_l_1.setZero();
                }
            }
    }
    for (const generic_custom_func &f : prob_->exprs(__pre_comp)) {
        f.custom_call((*shared_)[f]); ///< @todo pass update mode
    }
    bool no_eval = config != update_mode::eval_val && config != update_mode::eval_all;
    bool no_jac = config != update_mode::eval_jac && config != update_mode::eval_derivatives && config != update_mode::eval_all;
    bool no_hess = config != update_mode::eval_hess && config != update_mode::eval_derivatives && config != update_mode::eval_all;
    for_each<func_fields>([=, this](const generic_func &_f, func_approx_data &data) {
        _f.compute_approx(data,
                          !no_eval && _f.order() >= approx_order::zero,
                          !no_jac && _f.order() >= approx_order::first,
                          !no_hess && _f.order() >= approx_order::second);
    });
    if (update_cost) {
        inf_prim_res_ = 0.;
        for (const auto &field_data : dense_->approx_) {
            if (field_data.v_.size() == 0)
                continue; // skip empty fields
            inf_prim_res_ = std::max(field_data.v_.cwiseAbs().maxCoeff(), inf_prim_res_);
        }
        inf_comp_res_ = 0.;
        for (const auto &comp : dense_->comp_) {
            if (comp.size() == 0)
                continue; // skip empty fields
            inf_comp_res_ = std::max(comp.cwiseAbs().maxCoeff(), inf_comp_res_);
        }
        dense_->merit_ += dense_->cost_;
    }
}
void node_data::print_residuals() const {
    for (auto f : merit_data::stored_constr_fields) {
        fmt::println("Field {}: dim {} residual {}", field::name(f), dense_->approx_[f].v_.size(),
                     dense_->approx_[f].v_.transpose());
    }
}
namespace impl {
void data_mgr::create_data_batch(const ocp_ptr_t &prob, size_t N) {
    data_.try_emplace(prob->uid());
    auto &pool = data_[prob->uid()];
    std::lock_guard _lock(pool.mtx_);
    for (size_t i = 0; i < N; i++) {
        pool.emplace(maker_(prob));
    }
}

node_data_ptr_t data_mgr::get_data(const ocp_ptr_t &prob) {
    data_.try_emplace(prob->uid());
    auto &pool = data_[prob->uid()];
    std::lock_guard _lock(pool.mtx_);
    if (!pool.empty()) {
        auto p = std::move(pool.top());
        pool.pop();
        return p;
    } else {
        return nullptr;
    }
}

node_data *data_mgr::acquire(const ocp_ptr_t &prob) {
    auto p = get_data(prob);
    if (p) {
        return p.release();
    } else {
        return maker_(prob);
    }
}

node_data *data_mgr::acquire(const node_data *rhs) {
    return acquire(rhs->prob_);
}

void data_mgr::release(node_data *data) {
    auto &pool = data_[data->prob_->uid()];
    std::lock_guard _lock(pool.mtx_);
    pool.emplace(data);
}
} // namespace impl
} // namespace moto