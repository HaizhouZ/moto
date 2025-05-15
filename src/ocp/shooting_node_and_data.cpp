#include <atri/ocp/approx_storage.hpp>
#include <atri/ocp/shooting_node.hpp>
#include <ranges>

namespace atri {

/**
 * @brief for loop shortcut for funcs (dyn,cost,constr...)
 *
 * @param prob
 * @param callback [idx of field, idx of expr, pointer to expr]
 */
template <typename Callback>
inline void for_funcs(const problem_ptr_t &prob, Callback &&callback) {
    // loop with two variables due to the difference between idx and field no.
    for (size_t field : range_n(__dyn, field::num_func)) {
        size_t idx = 0;
        for (const auto &expr : prob->expr_[field]) {
            callback(field, idx++, static_cast<approx *>(expr.get()));
        }
    }
}

node_data::node_data(problem_ptr_t prob)
    : sym_(new sym_data(prob)), raw_(new approx_storage(prob)) {
    for_funcs(prob, [&](size_t field, [[maybe_unused]] size_t idx, approx *_f) {
        sparse_[field].push_back(_f->make_data(sym_, raw_));
    });
}

node_data::~node_data() {
    delete sym_;
    delete raw_;
}

void data_mgr::create_data_batch(problem_ptr_t prob, size_t N) {
    data_.try_emplace(prob->uid_);
    auto &pool = data_[prob->uid_];
    std::lock_guard _lock(pool.mtx_);
    for (size_t i = 0; i < N; i++) {
        pool.push(maker_(prob));
    }
}

node_data_ptr_t data_mgr::acquire_data(problem_ptr_t prob) {
    auto &pool = data_[prob->uid_];
    std::lock_guard _lock(pool.mtx_);
    if (!pool.empty()) {
        auto p = std::move(pool.top());
        pool.pop();
        return p;
    } else {
        return maker_(prob);
    }
}

void data_mgr::release_data(problem_ptr_t prob, node_data_ptr_t data) {
    auto &pool = data_[prob->uid_];
    std::lock_guard _lock(pool.mtx_);
    pool.push(std::move(data));
}

void shooting_node::swap(shooting_node &p) {
    problem_.swap(p.problem_);
    data_.swap(p.data_);
}

void shooting_node::update_approximation() {
    /// @todo: always eval residual?
    for_funcs(problem_,
              [this](size_t field, size_t idx_expr, approx *_f) {
                  _f->evaluate(*data_->sparse_[field][idx_expr],
                               true, _f->order() >= approx_order::first,
                               _f->order() >= approx_order::second);
              });
}

} // namespace atri