#ifndef MOTO_UTILS_CODEGEN_BIND_HPP
#define MOTO_UTILS_CODEGEN_BIND_HPP

#include <future>
#include <moto/core/expr.hpp>

namespace moto {
namespace utils {

std::future<void> generate_n_compile(expr_impl &func, const std::vector<sym> &in_args, const cs::SX &out,
                                     bool gen_eval = true, bool gen_jac = false, bool gen_hess = false);

} // namespace utils
} // namespace moto

#endif // MOTO_UTILS_CODEGEN_BIND_HPP