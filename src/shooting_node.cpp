#include <atri/ocp/shooting_node.hpp>
namespace atri {

/**
 * @brief for loop shortcut for funcs (dyn,cost,constr...)
 *
 * @param exprs
 * @param callback [idx of field, idx of expr, pointer to expr]
 */
template <typename callback_type>
inline void for_funcs(expr_sets_ptr_t exprs, callback_type &&callback) {
    // loop with two variables due to the difference between idx and field no.
    for (size_t field = __dyn; field != field::num; field++) {
        auto &_exprs = exprs->expr_[field];
        if (_exprs.empty()) {
            for (size_t idx_expr = 0; idx_expr < _exprs.size(); idx_expr++) {
                auto _f = std::static_pointer_cast<approx>(_exprs[idx_expr]);
                callback(field, idx_expr, _f);
            }
        }
    }
}

node_data::node_data(expr_sets_ptr_t exprs) : raw_data_(exprs) {
    for_funcs(exprs, [&](size_t field, size_t idx_expr, approx_ptr_t _f) {
        approx_[field].push_back(_f->make_data(raw_data_));
    });
}

void data_mgr::create_data_batch(expr_sets_ptr_t exprs, size_t N) {
    auto &pool = *data_[exprs->uid_];
    std::lock_guard _lock(pool.mtx_);
    for (size_t i = 0; i < N; i++)
        pool.push(maker_(exprs));
}

node_data_ptr_t data_mgr::acquire_data(expr_sets_ptr_t exprs) {
    auto &pool = *data_[exprs->uid_];
    std::lock_guard _lock(pool.mtx_);
    if (!pool.empty()) {
        auto p = std::move(pool.top());
        pool.pop();
        return p;
    } else {
        return maker_(exprs);
    }
}

void data_mgr::release_data(expr_sets_ptr_t exprs, node_data_ptr_t data) {
    auto &pool = *data_[exprs->uid_];
    std::lock_guard _lock(pool.mtx_);
    pool.push(data);
}

void shooting_node::swap(shooting_node &p) {
    expr_sets_.swap(p.expr_sets_);
    data_.swap(p.data_);
}

void shooting_node::update_approximation() {
    /// @todo: always eval residual?
    for_funcs(expr_sets_,
              [this](size_t field, size_t idx_expr, approx_ptr_t _f) {
                  _f->evaluate(expr_sets_, data_->approx_[field][idx_expr],
                               true, _f->order() >= approx_order::first,
                               _f->order() >= approx_order::second);
              });
}

} // namespace atri