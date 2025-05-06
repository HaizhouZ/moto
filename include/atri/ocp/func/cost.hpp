// a collection of cost functions;
// class cost : public approx{
// void add_cost
// }
#include <atri/ocp/core/approx.hpp>

namespace atri {
struct cost : public approx {
    cost(const std::string &name, approx_order order = approx_order::second) : approx(name, 1, __cost, order) {}
};
}; // namespace atri