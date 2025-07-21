#include <moto/utils/print.hpp>

namespace moto {
namespace utils {
void print_problem(const ocp_ptr_t &prob) {
    fmt::print("-------------------------------------------------\n");
    fmt::print("problem uid {}\n", prob->uid());
    for (size_t i = 0; i < field::num; i++) {
        if (prob->exprs(i).size() > 0) {
            fmt::print("field : {}, \ttotal dim {}\n",
                       field::name(static_cast<field_t>(i)),
                       prob->dim(i));
            for (const auto &expr : prob->exprs(i)) {
                fmt::print(" - {} dim: {}\n", expr->name(), expr->dim());
            }
        }
    }
    fmt::print("-------------------------------------------------\n");
}
} // namespace utils

} // namespace moto
