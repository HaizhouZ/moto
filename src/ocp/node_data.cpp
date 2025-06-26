#include <moto/ocp/shooting_node.hpp>
#include <ranges>

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
            callback(field, idx++, static_cast<func_impl &>(*expr));
        }
    }
}

node_data::node_data(const ocp_ptr_t &prob)
    : ocp_(prob), sym_(new sym_data(prob)), dense_(new approx_storage(prob)), shared_(new shared_data(prob, *sym_)) {
    for_funcs(prob, [&](size_t field, [[maybe_unused]] size_t idx, func_impl &_f) {
        sparse_[field].push_back(_f.make_approx_data_mapping(*sym_, *dense_, *shared_));
    });
}

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

node_data* data_mgr::acquire(const ocp_ptr_t &prob) {
    auto p = get_data(prob);
    if (p) {
        return p.release();
    } else {
        return maker_(prob);
    }
}

node_data* data_mgr::acquire(const node_data* rhs) {
    return acquire(rhs->ocp_);
}

void data_mgr::release(node_data* data) {
    auto &pool = data_[data->sym_->prob_->uid_];
    std::lock_guard _lock(pool.mtx_);
    pool.emplace(data);
}

void node_data::update_approximation() {
    /// @todo: always eval residual?
    // call to precompute
    for (const auto &expr : ocp_->expr_[__pre_comp]) {
        auto &f = static_cast<func_impl &>(*expr);
        f.call((*shared_)[f]);
    }
    for_funcs(ocp_,
              [this](size_t field, size_t idx_expr, func_impl &_f) {
                  _f.evaluate_approx(*sparse_[field][idx_expr],
                                      true, _f.order() >= approx_order::first,
                                      _f.order() >= approx_order::second);
              });
}

} // namespace moto