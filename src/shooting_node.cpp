#include <atri/ocp/shooting_node.hpp>
namespace atri {

/**
 * @brief for loop shortcut for funcs (dyn,cost,constr...)
 *
 * @param expr_collection
 * @param callback [idx of field, idx of expr, pointer to expr]
 */
template <typename callback_type>
inline void for_loop_funcs(expr_collection_ptr_t expr_collection,
                           callback_type&& callback) {
    for (size_t i = 0, field = field_type::dyn; field != field::num; i++, field++) {
        auto& _exprs = expr_collection->expr_[field];
        if (_exprs.empty()) {
            for (size_t j = 0; j < _exprs.size(); j++) {
                auto _c = std::static_pointer_cast<approximation>(_exprs[j]);
                callback(i, j, _c);
            }
        }
    }
}

stacked_approx_ptr mem_mgr::make_approx_data(expr_collection_ptr_t expr_collection) {
    stacked_approx_ptr d;
    for_loop_funcs(expr_collection, [&d](size_t i, size_t j, approximation_ptr_t _c) {
        d[i].push_back(_c->make_approx_data());
    });
    return d;
}

void mem_mgr::create_data_batch(expr_collection_ptr_t expr_collection, size_t N) {
    auto& _approx = *approx_[expr_collection->uid_];
    std::lock_guard _lock(_approx.mtx_);
    for (size_t i = 0; i < N; i++)
        _approx.data_.push(make_approx_data(expr_collection));
}

stacked_approx_ptr mem_mgr::acquire_approx(expr_collection_ptr_t expr_collection) {
    auto& _approx = *approx_[expr_collection->uid_];
    std::lock_guard _lock(_approx.mtx_);
    if (!_approx.data_.empty()) {
        auto p = std::move(_approx.data_.top());
        _approx.data_.pop();
        return p;
    } else {
        return make_approx_data(expr_collection);
    }
}

void mem_mgr::release_approx(expr_collection_ptr_t expr_collection, stacked_approx_ptr&& data) {
    auto& _approx = *approx_[expr_collection->uid_];
    std::lock_guard _lock(_approx.mtx_);
    _approx.data_.push(std::move(data));
}

void shooting_node::swap(shooting_node& p) {
    expr_collection_.swap(p.expr_collection_);
    approx_.swap(p.approx_);
    primal_data_.swap(p.primal_data_);
}

void shooting_node::collect_data() {
    for_loop_funcs(expr_collection_, [this](size_t field, size_t j, approximation_ptr_t _c) {
        if (_c->approx_level() == approx_type::first) {
            _c->evaluate<true, true>(expr_collection_, primal_data_, approx_[field][j]);
        }
    });
}

}  // namespace atri