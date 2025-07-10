#ifndef __MOTO_PROBLEM_HPP__
#define __MOTO_PROBLEM_HPP__

#include <array>
#include <map>
#include <unordered_map>
#include <vector>

#include <moto/core/expr.hpp>

namespace moto {
/**
 * @brief problem formulation of an OCP stage
 *
 */
class ocp {
  protected:
    inline static size_t max_uid = 0;
    ocp() : uid_(max_uid++) {}
    ocp(const ocp &rhs)
        : uid_(max_uid++), expr_(rhs.expr_), d_idx_(rhs.d_idx_),
          pos_by_uid_(rhs.pos_by_uid_), dim_(rhs.dim_) {}
    bool add_impl(const expr_ptr_t &expr);

  public:
    const size_t uid_ = 0;
    /// collection of all expressions in the problem
    std::array<std::vector<expr_ptr_t>, field::num> expr_;
    /// data index of expr in serialized vector, by uid
    std::unordered_map<size_t, std::pair<size_t, size_t>> d_idx_;
    /// position of expr in expr_ e.g., no. xxx, by uid
    std::unordered_map<size_t, size_t> pos_by_uid_;
    /// dimension of each field
    std::array<size_t, field::num> dim_{};

    static auto make() { return std::shared_ptr<ocp>(new ocp()); }
    auto copy() { return std::shared_ptr<ocp>(new ocp(*this)); }

    scalar_t *get_data_ptr(scalar_t *data, expr_impl &expr) const {
        return data + get_expr_start(expr);
    }
    scalar_t *get_data_ptr(scalar_t *data, expr_impl &expr, size_t offset) const {
        return data + get_expr_start(expr) * offset;
    }
    vector_ref extract(vector_ref data, const expr_impl &expr) const {
        return data.segment(get_expr_start(expr), expr.dim_);
    }
    /**
     * @brief add expr to problem formulation
     * @note will copy the shared pointer
     * @param expr expression to be added
     */
    void add(const expr_ptr_t &expr);
    /**
     * @brief add expr to problem formulation
     * @note will move the shared pointer
     * @param expr expression to be added
     */
    void add(expr_ptr_t &&expr);

    template <typename derived>
        requires std::is_base_of_v<expr_impl, derived>
    void add(const std::vector<std::shared_ptr<derived>> &exprs) {
        for (const auto &expr_ : exprs) {
            add(expr_);
        }
    }

    template <typename derived>
        requires std::is_base_of_v<expr_impl, derived>
    void add(std::vector<std::shared_ptr<derived>> &&exprs) {
        for (auto &expr_ : exprs) {
            add(std::move(expr_));
        }
    }

    /**
     * @brief get start index of expr in its field
     */
    template <typename derived>
        requires std::is_base_of_v<expr_impl, derived>
    inline size_t get_expr_start(const std::shared_ptr<derived> &expr) const {
        return get_expr_start(*expr);
    }
    size_t get_expr_start(const expr_impl &expr) const {
        try {
            return d_idx_.at(expr.uid_).first;
        } catch (const std::exception &e) {
            throw std::runtime_error(fmt::format("expr {} uid {} cannot be found", expr.name_, expr.dim_));
        }
    }
};

def_ptr(ocp);

/**
 * @brief backward copy from next stacked y to current x, usually to ensure consistency of initialization
 *
 * @param _y
 * @param _x
 * @param prob_cur
 * @param prob_next
 */
inline void copy_y_to_x(vector_ref from_y, vector_ref to_x,
                        const ocp_ptr_t &prob_y, const ocp_ptr_t &prob_x) {
    for (auto &y : prob_y->expr_[__y]) {
        prob_x->extract(to_x, *expr_lookup::get<sym>(y->uid_ - 1)) = prob_y->extract(from_y, *y);
    }
}
/**
 * @brief forward copy from stacked x to y
 *
 * @param _x
 * @param _y
 * @param prob_next
 * @param prob_cur
 */
inline void copy_x_to_y(vector_ref from_x, vector_ref to_y,
                        const ocp_ptr_t &prob_x, const ocp_ptr_t &prob_y) {
    for (auto &x : prob_x->expr_[__x]) {
        prob_y->extract(to_y, *expr_lookup::get<sym>(x->uid_ + 1)) = prob_x->extract(from_x, *x);
    }
}

/**
 * @brief Get the permutation matrix that maps y to x, i.e., X = P * Y
 * @example for derivative conversion: dfdy = dfdx * P
 * @param prob_y problem of which the y field (order of syms) is used
 * @param prob_x problem of which the x field (order of syms) is used
 * @return Eigen::PermutationMatrix<-1, -1>& the permutation matrix that maps y to x
 */
Eigen::PermutationMatrix<-1, -1> &permutation_from_y_to_x(const ocp_ptr_t &prob_y, const ocp_ptr_t &prob_x);
} // namespace moto

#endif /*__problem_FORMULATION_*/