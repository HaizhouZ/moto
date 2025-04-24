#include <atri/ocp/shooting_node.hpp>
namespace atri {

/**
 * @brief for loop shortcut for funcs (dyn,cost,constr...)
 *
 * @param expr_sets
 * @param callback [idx of field, idx of expr, pointer to expr]
 */
template <typename callback_type>
inline void for_funcs(expr_sets_ptr_t expr_sets, callback_type &&callback) {
    // loop with two variables due to the difference between idx and field no.
    for (size_t field = field::type::dyn; field != field::num; field++) {
        auto &_exprs = expr_sets->expr_[field];
        if (_exprs.empty()) {
            for (size_t idx_expr = 0; idx_expr < _exprs.size(); idx_expr++) {
                auto _c = std::static_pointer_cast<approx>(_exprs[idx_expr]);
                callback(field, idx_expr, _c);
            }
        }
    }
}

approx_sets_data data_mgr::make_data(expr_sets_ptr_t expr_sets) {
    approx_sets_data d;
    for_funcs(expr_sets, [&d](size_t field, size_t idx_expr, approx_ptr_t _c) {
        d[field].push_back(_c->make_data());
    });
    return d;
}

void data_mgr::create_data_batch(expr_sets_ptr_t expr_sets, size_t N) {
    auto &_approx = *approx_[expr_sets->uid_];
    std::lock_guard _lock(_approx.mtx_);
    for (size_t i = 0; i < N; i++)
        _approx.data_.push(make_data(expr_sets));
}

approx_sets_data data_mgr::acquire_data(expr_sets_ptr_t expr_sets) {
    auto &_approx = *approx_[expr_sets->uid_];
    std::lock_guard _lock(_approx.mtx_);
    if (!_approx.data_.empty()) {
        auto p = std::move(_approx.data_.top());
        _approx.data_.pop();
        return p;
    } else {
        return make_data(expr_sets);
    }
}

void data_mgr::release_data(expr_sets_ptr_t expr_sets,
                            approx_sets_data &&data) {
    auto &_approx = *approx_[expr_sets->uid_];
    std::lock_guard _lock(_approx.mtx_);
    _approx.data_.push(std::move(data));
}

void shooting_node::swap(shooting_node &p) {
    expr_sets_.swap(p.expr_sets_);
    approx_.swap(p.approx_);
    primal_data_.swap(p.primal_data_);
}

void shooting_node::update_approximation() {
    for_funcs(expr_sets_,
              [this](size_t field, size_t idx_expr, approx_ptr_t _c) {
                  if (_c->order() == approx_order::first) {
                      _c->evaluate<true, true>(expr_sets_, primal_data_,
                                               approx_[field][idx_expr]);
                  }
              });
}

} // namespace atri