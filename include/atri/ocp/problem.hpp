#ifndef __problem_FORMULATION__
#define __problem_FORMULATION__

#include <array>
#include <map>
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
    std::array<std::vector<expr_ptr_t>, field::num> expr_;
    std::unordered_map<size_t, std::pair<size_t, size_t>> d_idx_; // data index
    struct expr_info {
        expr *p;
        size_t pos;
    };
    std::map<std::string, expr_info, std::less<>> by_name_;
    std::unordered_map<size_t, size_t> pos_by_uid_;
    std::array<size_t, field::num> dim_{};

    problem() : uid_(max_uid++) {}
    problem(const problem &rhs)
        : uid_(max_uid++), expr_(rhs.expr_), d_idx_(rhs.d_idx_),
          by_name_(rhs.by_name_), pos_by_uid_(rhs.pos_by_uid_), dim_(rhs.dim_) {}

    auto copy() { return std::make_shared<problem>(*this); }

    scalar_t *get_data_ptr(scalar_t *data, expr &expr) const {
        return data + get_expr_start(expr);
    }
    scalar_t *get_data_ptr(scalar_t *data, expr &expr, size_t offset) const {
        return data + get_expr_start(expr) * offset;
    }
    /**
     * @brief add expr to problem formulation
     *
     * @param expr expression to be added
     */
    void add(expr_ptr_t expr) {
        size_t _uid = expr->uid_;
        if (d_idx_.find(_uid) == d_idx_.end()) { // skip repeated
            size_t &n0 = dim_[expr->field_];
            size_t n1 = n0 + expr->dim_;
            expr_[expr->field_].push_back(expr);
            d_idx_[_uid] = std::make_pair(n0, n1);
            n0 = d_idx_[_uid].second;
            by_name_.try_emplace(expr->name_, expr_info{expr.get(), expr_[expr->field_].size() - 1});
            pos_by_uid_.try_emplace(_uid, expr_[expr->field_].size() - 1);
            const auto &aux = expr->get_aux();
            if (!aux.empty()) {
                add(aux);
            }
        }
    }

    template <typename derived, typename std::enable_if_t<std::is_base_of_v<expr, derived>, int> = 0>
    void add(const std::vector<std::shared_ptr<derived>> &exprs) {
        for (auto expr_ : exprs) {
            add(expr_);
        }
    }

    template <typename derived, typename std::enable_if_t<std::is_base_of_v<expr, derived>, int> = 0>
    void add(std::shared_ptr<derived> rhs) {
        add(std::static_pointer_cast<expr>(rhs));
    }

    void add(const collection &rhs) { add(rhs.get()); }

    /**
     * @brief get start index of expr in its field
     */
    size_t get_expr_start(const expr_ptr_t& expr) const {
        return d_idx_.at(expr->uid_).first;
    }
    size_t get_expr_start(expr &expr) const {
        return d_idx_.at(expr.uid_).first;
    }
};

def_ptr(problem);
} // namespace atri

#endif /*__problem_FORMULATION_*/