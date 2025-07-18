#include <moto/ocp/impl/data_mgr.hpp>

namespace moto {

/**
 * @brief for loop shortcut for funcs (dyn,cost,constr...)
 *
 * @param prob
 * @param callback [idx of field, idx of expr, pointer to expr]
 */
template <typename Callback>
inline void for_funcs(const ocp_ptr_t &prob, Callback &&callback) {
    // loop with two variables due to the difference between idx and field no.
    for (size_t field : range_n(__dyn, field::num_func)) {
        size_t idx = 0;
        for (const auto &expr : prob->expr_[field]) {
            callback(field, idx++, static_cast<impl::func &>(*expr));
        }
    }
}

node_data::node_data(const ocp_ptr_t &prob)
    : prob_(prob), sym_(new sym_data(prob)), dense_(new approx_storage(prob)), shared_(new shared_data(prob, *sym_)) {
    for_funcs(prob, [&](size_t field, [[maybe_unused]] size_t idx, impl::func &_f) {
        sparse_[field].push_back(_f.make_approx_map(*sym_, *dense_, *shared_));
    });
}
void node_data::clear_merit_jac() {
    // set cost jacobian to zero
    for (auto field : primal_fields) {
        dense_->jac_[field].setZero();
        dense_->jac_modification_[field].setZero();
    }
}
void node_data::clear_merit_hessian() {
    // set hessian to zero
    for (auto &hess_l_0 : dense_->hessian_) {
        for (auto &hess_l_1 : hess_l_0) {
            hess_l_1.setZero();
        }
    }
}
void node_data::update_approximation(bool eval_only) {
    /// @todo: always eval residual?
    // call to precompute
    dense_->cost_ = 0.;
    dense_->merit_ = 0.;
    clear_merit_jac();
    clear_merit_hessian();
    for (const auto &expr : prob_->expr_[__pre_comp]) {
        auto &f = static_cast<impl::func &>(*expr);
        f.call((*shared_)[f]);
    }
    for_funcs(prob_,
              [this, eval_only](size_t field, size_t idx_expr, impl::func &_f) {
                  _f.evaluate_approx(*sparse_[field][idx_expr], true,
                                     !eval_only && _f.order() >= approx_order::first,
                                     !eval_only && _f.order() >= approx_order::second);
              });
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

void node_data::merge_jacobian_modification() {
    for (const auto &field : primal_fields) {
        dense_->jac_[field] += dense_->jac_modification_[field];
    }
}

namespace impl {
void data_mgr::create_data_batch(const ocp_ptr_t &prob, size_t N) {
    data_.try_emplace(prob->uid_);
    auto &pool = data_[prob->uid_];
    std::lock_guard _lock(pool.mtx_);
    for (size_t i = 0; i < N; i++) {
        pool.emplace(maker_(prob));
    }
}

node_data_ptr_t data_mgr::get_data(const ocp_ptr_t &prob) {
    auto &pool = data_[prob->uid_];
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
    auto &pool = data_[data->sym_->prob_->uid_];
    std::lock_guard _lock(pool.mtx_);
    pool.emplace(data);
}
} // namespace impl
} // namespace moto