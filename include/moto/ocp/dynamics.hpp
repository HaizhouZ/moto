#ifndef MOTO_TRAITS_DYNAMICS_HPP
#define MOTO_TRAITS_DYNAMICS_HPP
#include <moto/ocp/problem.hpp>

namespace moto {

struct dynamics {
    static auto make_input(const std::string &name, size_t dim) {
        return sym(name, dim, __u);
    }
    static auto make_param(const std::string &name, size_t dim) {
        return sym(name, dim, __p);
    }
    static auto make_state(const std::string &name, size_t dim) {
        auto temp = sym(name, dim, __x);
        auto next = sym(name + "_nxt", dim, __y);
        return std::make_pair(temp, next);
    }
    /**
     * @brief backward copy from next stacked y to current x, usually to ensure consistency of initialization
     *
     * @param _y
     * @param _x
     * @param prob_cur
     * @param prob_next
     */
    static void copy_y_to_x(vector_ref from_y, vector_ref to_x,
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
    static void copy_x_to_y(vector_ref from_x, vector_ref to_y,
                            const ocp_ptr_t &prob_x, const ocp_ptr_t &prob_y) {
        for (auto &x : prob_x->expr_[__x]) {
            prob_y->extract(to_y, *expr_lookup::get<sym>(x->uid_ + 1)) = prob_x->extract(from_x, *x);
        }
    }
};
} // namespace moto

#endif // MOTO_TRAITS_DYNAMICS_HPP