#include <atri/ocp/problem.hpp>

namespace atri {
void problem::add_impl(const expr_ptr_t &expr) {
    size_t _uid = expr->uid_;
    if (d_idx_.find(_uid) == d_idx_.end()) { // skip repeated
        if (!expr->finalize()) {
            throw std::runtime_error(fmt::format("cannot finalize expr {} uid {}", expr->name_, expr->uid_));
        }
        size_t &n0 = dim_[expr->field_];
        size_t n1 = n0 + expr->dim_;
        d_idx_[_uid] = std::make_pair(n0, n1);
        n0 = d_idx_[_uid].second;
        pos_by_uid_.try_emplace(_uid, expr_[expr->field_].size() - 1);
        const auto &aux = expr->get_aux();
        if (!aux.empty()) {
            add(aux);
        }
    }
}
void problem::add(const expr_ptr_t &expr) {
    expr_[expr->field_].push_back(expr);
    add_impl(expr_[expr->field_].back());
}
void problem::add(expr_ptr_t &&expr) {
    add_impl(expr_[expr->field_].emplace_back(std::move(expr)));
}
size_t problem::get_expr_start(const expr &expr) const {
    try {
        return d_idx_.at(expr.uid_).first;
    } catch (const std::exception &e) {
        throw std::runtime_error(fmt::format("expr {} uid {} cannot be found", expr.name_, expr.dim_));
    }
}
} // namespace atri
