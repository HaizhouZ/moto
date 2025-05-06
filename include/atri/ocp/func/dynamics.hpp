#ifndef ATRI_TRAITS_DYNAMICS_HPP
#define ATRI_TRAITS_DYNAMICS_HPP
#include <atri/core/expr.hpp>

namespace atri {

struct dynamics {
  protected:
    static auto make_input(const std::string &name, size_t dim) {
        return std::make_shared<sym>(name, dim, __u);
    }
    static auto make_state(const std::string &name, size_t dim) {
        auto temp = std::make_shared<sym>(name, dim, __x);
        auto next = std::make_shared<sym>(name + "_nxt", dim, __y);
        return std::make_pair(temp, next);
    }
};
} // namespace atri

#endif // ATRI_TRAITS_DYNAMICS_HPP