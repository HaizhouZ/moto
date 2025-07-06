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
  private:
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
    vector_ref extract(vector_ref data, const expr_impl& expr) const {
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
     * @brief add a expr_list to the problem
     * will be called if argument is a list of raw pointers
     * @param exprs
     */
    void add(expr_list &&exprs) {
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
    size_t get_expr_start(const expr_impl &expr) const;
};

def_ptr(ocp);
} // namespace moto

#endif /*__problem_FORMULATION_*/