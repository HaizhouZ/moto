#ifndef __PROBLEM_FORMULATION__
#define __PROBLEM_FORMULATION__

#include <vector>
#include <map>

#include <atri/core/expression_base.hpp>

namespace atri {
/**
 * @brief problem formulation of an OCP stage
 *
 */
struct problem {
    static size_t max_uid;
    const size_t uid_ = 0;
    // std::map<field_type, std::map<size_t, expr_ptr_t>> expr_;
    std::vector<expr_ptr_t> expr_[field::num];
    std::map<size_t, std::pair<size_t, size_t>> idx_;
    size_t dim_[field::num] = {0};

    problem()
        : uid_(max_uid++) {}

    const scalar_t* get_data_ptr(const scalar_t* data, expr_ptr_t expr) {
        return data + get_expr_idx(expr).first;
    }

    /**
     * @brief add expr to problem formulation
     *
     * @param expr expression to be added
     * @param field in [field_type]
     */
    void add_expr(expr_ptr_t expr) {
        size_t _uid = expr->uid_;
        size_t& n0 = dim_[expr->field_];
        size_t n1 = n0 + expr->dim_;
        expr_[expr->field_].push_back(expr);
        idx_[_uid] = std::make_pair(n0, n1);
        n0 = idx_[_uid].second;
    }

    /**
     * @brief get the idx of an expr named by [name]
     *
     * @param expr expression to look up
     * @param field type of the expression
     * @return std::pair<size_t, size_t> [start, end) of the expression
     */
    std::pair<size_t, size_t> get_expr_idx(expr_ptr_t expr) {
        try {
            return idx_[expr->uid_];
        } catch (const std::exception& e) {
            throw std::runtime_error(fmt::format(
                "Cannot get idx of no.{}:{}", expr->name_, expr->uid_));
        }
    }
};

def_ptr(problem);
}  // namespace atri

#endif /*__PROBLEM_FORMULATION_*/