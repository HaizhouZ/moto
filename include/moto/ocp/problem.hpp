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
    ocp() = default; // default constructor
    ocp(const ocp &rhs)
        : expr_(rhs.expr_), d_idx_(rhs.d_idx_),
          pos_by_uid_(rhs.pos_by_uid_), dim_(rhs.dim_) {}
    bool add_impl(expr *);

    uid_t<ocp> uid_;
    /// collection of all expressions in the problem
    std::array<expr_list, field::num> expr_;
    /// data index of expr in serialized vector, by uid
    std::unordered_map<size_t, std::pair<size_t, size_t>> d_idx_;
    /// position of expr in expr_ e.g., no. xxx, by uid
    std::unordered_map<size_t, size_t> pos_by_uid_;
    /// dimension of each field
    std::array<size_t, field::num> dim_{};

  public:
    CONST_ATTR_GETTER(uid);                                                  ///< getter for uid
    const auto &exprs(size_t f) const { return expr_.at(f); }                ///< getter for expr_
    const auto &pos(const expr *e) const { return pos_by_uid_.at(e->uid()); } ///< getter for pos_by_uid_
    size_t dim(size_t f) const { return dim_.at(f); }                        ///< getter for dim_

    static auto create() { return std::shared_ptr<ocp>(new ocp()); }
    auto clone() { return std::shared_ptr<ocp>(new ocp(*this)); }

    scalar_t *get_data_ptr(scalar_t *data, expr *expr) const {
        return data + get_expr_start(expr);
    }
    scalar_t *get_data_ptr(scalar_t *data, expr *expr, size_t offset) const {
        return data + get_expr_start(expr) * offset;
    }
    vector_ref extract(vector_ref data, const expr *expr) const {
        return data.segment(get_expr_start(expr), expr->dim());
    }
    /**
     * @brief add expr to problem formulation
     * @note will copy the shared pointer
     * @param expr expression to be added
     */
    void add(expr *expr);

    template <typename derived = expr>
        requires std::is_base_of_v<expr, derived>
    void add(const std::initializer_list<derived *> &exprs) {
        for (auto &expr_ : exprs) {
            add(expr_);
        }
    }

    template <typename derived>
        requires std::is_base_of_v<expr, derived>
    void add(const std::vector<derived *> &exprs) {
        for (auto &expr_ : exprs) {
            add(expr_);
        }
    }

    /**
     * @brief get start index of expr in its field
     */
    size_t get_expr_start(const expr *expr) const {
        try {
            return d_idx_.at(expr->uid()).first;
        } catch (const std::exception &e) {
            throw std::runtime_error(fmt::format("expr {} uid {} cannot be found", expr->name(), expr->dim()));
        }
    }
};

def_ptr(ocp);

/**
 * @brief backward copy from next stacked y to current x, usually to ensure consistency of initialization
 */
inline void copy_y_to_x(vector_ref from_y, vector_ref to_x,
                        const ocp_ptr_t &prob_y, const ocp_ptr_t &prob_x) {
    for (auto &y : prob_y->exprs(__y)) {
        prob_x->extract(to_x, expr_lookup::get(y->uid() - 1)) = prob_y->extract(from_y, y);
    }
}
/**
 * @brief forward copy from stacked x to y
 */
inline void copy_x_to_y(vector_ref from_x, vector_ref to_y,
                        const ocp_ptr_t &prob_x, const ocp_ptr_t &prob_y) {
    for (auto &x : prob_x->exprs(__x)) {
        prob_y->extract(to_y, expr_lookup::get(x->uid() + 1)) = prob_x->extract(from_x, x);
    }
}

/**
 * @brief Get the permutation matrix that maps y to x, i.e., X = P * Y.
 * for example: derivative conversion: dfdy = dfdx * P
 * @param prob_y problem of which the y field (order of syms) is used
 * @param prob_x problem of which the x field (order of syms) is used
 * @return Eigen::PermutationMatrix<-1, -1>& the permutation matrix that maps y to x
 */
Eigen::PermutationMatrix<-1, -1> &permutation_from_y_to_x(const ocp_ptr_t &prob_y, const ocp_ptr_t &prob_x);
} // namespace moto

#endif // __MOTO_PROBLEM_HPP__