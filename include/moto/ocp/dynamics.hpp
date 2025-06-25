#ifndef MOTO_TRAITS_DYNAMICS_HPP
#define MOTO_TRAITS_DYNAMICS_HPP
#include <moto/core/expr.hpp>

namespace moto {

struct dynamics {
    static auto make_input(const std::string &name, size_t dim) {
        return sym(name, dim, __u);
    }
    static auto make_state(const std::string &name, size_t dim) {
        auto temp = sym(name, dim, __x);
        auto next = sym(name + "_nxt", dim, __y);
        return std::make_pair(temp, next);
    }
};
} // namespace moto

#endif // MOTO_TRAITS_DYNAMICS_HPP