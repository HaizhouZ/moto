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

Eigen::PermutationMatrix<-1, -1> &permutation_from_y_to_x(const ocp_ptr_t &prob_y, const ocp_ptr_t &prob_x) {
    using perm_type = Eigen::PermutationMatrix<-1, -1>;
    static std::unordered_map<size_t, std::unordered_map<size_t, perm_type>> perm_cache;
    assert(prob_y->dim_[__y] == prob_x->dim_[__x] && "dimension between states must match!");
    perm_cache.try_emplace(prob_y->uid_);
    auto [it, inserted] = perm_cache[prob_y->uid_].try_emplace(prob_x->uid_, prob_x->dim_[__x]);
    if (!inserted)
        return it->second; // already exists
    else {
        auto &perm = it->second;
        size_t col_y = 0;
        for (auto &y : prob_y->expr_[__y]) {
            auto &x = expr_lookup::get<sym>(y->uid_ - 1);
            size_t x0 = prob_x->get_expr_start(*x);
            size_t x1 = x0 + x->dim_;
            for (size_t i : range(x0, x1)) {
                perm.indices()[col_y++] = i;
            }
        }
        return perm;
    }
}
} // namespace moto
