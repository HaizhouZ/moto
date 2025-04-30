#ifndef __problem_FORMULATION__
#define __problem_FORMULATION__

#include <unordered_map>
#include <vector>

#include <atri/core/expr.hpp>

namespace atri {
/**
 * @brief problem formulation of an OCP stage
 *
 */
struct problem {
    static size_t max_uid;
    const size_t uid_ = 0;
    std::vector<expr_ptr_t> expr_[field::num];
    std::unordered_map<size_t, std::pair<size_t, size_t>> d_idx_;
    size_t dim_[field::num] = {0};

    problem() : uid_(max_uid++) {}

    scalar_t *get_data_ptr(scalar_t *data, expr_ptr_t expr) const {
        return data + get_expr_start(expr);
    }
    scalar_t *get_data_ptr(scalar_t *data, expr_ptr_t expr,
                           size_t offset) const {
        return data + get_expr_start(expr) * offset;
    }
    /**
     * @brief add expr to problem formulation
     *
     * @param expr expression to be added
     * @param field in [field_t]
     */
    void add(expr_ptr_t expr) {
        size_t _uid = expr->uid_;
        size_t &n0 = dim_[expr->field_];
        size_t n1 = n0 + expr->dim_;
        expr_[expr->field_].push_back(expr);
        d_idx_[_uid] = std::make_pair(n0, n1);
        n0 = d_idx_[_uid].second;
    }

    template <typename derived>
    void add(const std::vector<std::shared_ptr<derived>> &exprs) {
        static_assert(std::is_base_of_v<expr, derived>);
        for (auto expr_ : exprs) {
            add(expr_);
        }
    }

    /**
     * @brief get the idx of an expr named by [name]
     *
     * @param expr expression to look up
     * @param field type of the expression
     * @return std::pair<size_t, size_t> [start, length) of the expression
     */
    size_t get_expr_start(expr_ptr_t expr) const {
        return d_idx_.at(expr->uid_).first;
    }
    size_t get_expr_start(expr &expr) const {
        return d_idx_.at(expr.uid_).first;
    }
};

def_ptr(problem);
} // namespace atri

#endif /*__problem_FORMULATION_*/