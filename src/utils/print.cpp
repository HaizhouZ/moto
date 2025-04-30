#include <atri/utils/print.hpp>

namespace atri {
namespace utils {
void print_problem(problem_ptr_t prob) {
    for (int i = 0; i < field::num; i++) {
        fmt::print("field : {}, \ttotal dim {}\n",
                   magic_enum::enum_name(static_cast<field_t>(i)),
                   prob->dim_[i]);
        for (auto expr : prob->expr_[i]) {
            fmt::print(" - {} dim: {}\n", expr->name_, expr->dim_);
        }
    }
}
} // namespace utils

} // namespace atri
