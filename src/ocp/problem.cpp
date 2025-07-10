#include <moto/ocp/problem.hpp>

namespace moto {
bool ocp::add_impl(const expr_ptr_t &expr) {
    size_t _uid = expr->uid_;
    if (d_idx_.find(_uid) == d_idx_.end()) { // skip repeated
        if (!expr->finalize()) {
            throw std::runtime_error(fmt::format("cannot finalize expr {} uid {}", expr->name_, expr->uid_));
        }
        size_t &n0 = dim_[expr->field_];
        size_t n1 = n0 + expr->dim_;
        d_idx_[_uid] = std::make_pair(n0, n1);
        // fmt::print("add expr {} uid {} field {} dim {} to problem {} n0 {}, n1 {}\n",
        //            expr->name_, _uid, magic_enum::enum_name(expr->field_), expr->dim_, uid_, n0, n1);
        n0 = d_idx_[_uid].second;
        pos_by_uid_.try_emplace(_uid, expr_[expr->field_].size());
        auto &dep = expr->get_dep();
        if (!dep.empty()) {
            add(dep);
        }
        return true;
    }
    return false;
}
void ocp::add(const expr_ptr_t &expr) {
    if (add_impl(expr))
        expr_[expr->field_].push_back(expr);
}
void ocp::add(expr_ptr_t &&expr) {
    if (add_impl(expr))
        expr_[expr->field_].emplace_back(std::move(expr));
}
} // namespace moto
