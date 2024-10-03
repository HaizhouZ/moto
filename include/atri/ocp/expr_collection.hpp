#ifndef __expr_collection_FORMULATION__
#define __expr_collection_FORMULATION__

#include <vector>
#include <unordered_map>

#include <atri/core/expr.hpp>

namespace atri {
/**
 * @brief expr_collection formulation of an OCP stage
 *
 */
struct expr_collection {
    static size_t max_uid;
    const size_t uid_ = 0;
    // std::map<field_type, std::map<size_t, expr_ptr_t>> expr_;
    std::vector<expr_ptr_t> expr_[field::num];
    std::unordered_map<size_t, std::pair<size_t, size_t>> idx_;
    size_t dim_[field::num] = {0};

    expr_collection()
        : uid_(max_uid++) {}

    const scalar_t* get_data_ptr(const scalar_t* data, expr_ptr_t expr) {
        return data + get_expr_idx(expr).first;
    }

    /**
     * @brief add expr to expr_collection formulation
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

def_ptr(expr_collection);
}  // namespace atri

#endif /*__expr_collection_FORMULATION_*/