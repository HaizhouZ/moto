#ifndef __expr_sets_FORMULATION__
#define __expr_sets_FORMULATION__

#include <unordered_map>
#include <vector>

#include <atri/core/expr.hpp>

namespace atri {
/**
 * @brief expr_sets formulation of an OCP stage
 *
 */
struct expr_sets {
    static size_t max_uid;
    const size_t uid_ = 0;
    std::vector<expr_ptr_t> expr_[field::num];
    std::unordered_map<size_t, size_t> idx_;
    size_t dim_[field::num] = {0};

    expr_sets() : uid_(max_uid++) {}

    /**
     * @brief add expr to expr_sets formulation
     *
     * @param expr expression to be added
     * @param field in [field::type]
     */
    void add(expr_ptr_t expr) {
        size_t _uid = expr->uid_;
        size_t &n0 = dim_[expr->field_];
        size_t n1 = n0 + expr->dim_;
        idx_[expr->uid_] = expr_[expr->field_].size();
        expr_[expr->field_].push_back(expr);
    }

};

def_ptr(expr_sets);
} // namespace atri

#endif /*__expr_sets_FORMULATION_*/