#include <moto/ocp/problem.hpp>

namespace moto {
INIT_UID_(ocp);
bool ocp::add_impl(expr *expr) {
    size_t _uid = expr->uid();
    if (d_idx_.find(_uid) == d_idx_.end()) { // skip repeated
        // add dependencies
        auto &dep = expr->dep();
        if (!dep.empty()) {
            add(dep);
        }
        if (!expr->finalize()) {
            throw std::runtime_error(fmt::format("cannot finalize expr {} uid {}", expr->name(), expr->uid()));
        }
        size_t &n0 = dim_[expr->field()];
        size_t n1 = n0 + expr->dim();
        d_idx_[_uid] = std::make_pair(n0, n1);
        // fmt::print("add expr {} uid {} field {} dim {} to problem {} n0 {}, n1 {}\n",
        //            expr->name(), _uid, field::name(expr->field()), expr->dim(), uid_, n0, n1);
        n0 = d_idx_[_uid].second;
        pos_by_uid_.try_emplace(_uid, expr_[expr->field()].size());
        return true;
    }
    return false;
}
void ocp::add(expr *expr) {
    if (add_impl(expr))
        expr_[expr->field()].push_back(expr);
}

Eigen::PermutationMatrix<-1, -1> &permutation_from_y_to_x(const ocp_ptr_t &prob_y, const ocp_ptr_t &prob_x) {
    using perm_type = Eigen::PermutationMatrix<-1, -1>;
    static std::unordered_map<size_t, std::unordered_map<size_t, perm_type>> perm_cache;
    assert(prob_y->dim(__y) == prob_x->dim(__x) && "dimension between states must match!");
    perm_cache.try_emplace(prob_y->uid());
    auto [it, inserted] = perm_cache[prob_y->uid()].try_emplace(prob_x->uid(), prob_x->dim(__x));
    if (!inserted)
        return it->second; // already exists
    else {
        auto &perm = it->second;
        size_t col_y = 0;
        for (auto &y : prob_y->exprs(__y)) {
            auto x = expr_lookup::get(y->uid() - 1);
            size_t x0 = prob_x->get_expr_start(x);
            size_t x1 = x0 + x->dim();
            for (size_t i : range(x0, x1)) {
                perm.indices()[col_y++] = i;
            }
        }
        return perm;
    }
}
} // namespace moto
