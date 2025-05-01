#ifndef DOUBLE_INTEGRATOR_COST_HPP
#define DOUBLE_INTEGRATOR_COST_HPP

#include <atri/ocp/core/approx.hpp>

namespace atri {

struct cost : public approx {
    cost(const std::string &name, approx_order order = approx_order::second) : approx(name, 1, __cost, order) {}
};
template <typename T>
struct shared : public std::shared_ptr<T> {
    shared() : std::shared_ptr<T>(std::make_shared<T>()) {}
};

struct doubleIntegratorCost : public cost {
    doubleIntegratorCost() : cost("doubleIntegratorCost") {}
};

} // namespace atri

#endif // DOUBLE_INTEGRATOR_COST_HPP