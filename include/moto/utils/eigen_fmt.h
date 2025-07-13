#ifndef __MOTO_FMT_EIGEN__
#define __MOTO_FMT_EIGEN__

#include <Eigen/Core>
#include <charconv>
#include <fmt/core.h>
#include <fmt/format.h>
#include <sstream>

namespace fmt {
template <typename derived>
    requires std::is_base_of_v<Eigen::MatrixBase<derived>, derived>
struct formatter<derived> {
    int precision = 2;

    constexpr auto parse(format_parse_context &ctx) -> format_parse_context::iterator {
        auto it = ctx.begin();
        auto end = ctx.end();

        if (it != end && *it == '.') {
            ++it;
            int value = 0;
            while (it != end && *it >= '0' && *it <= '9') {
                value = value * 10 + (*it - '0');
                ++it;
            }
            precision = value;
        }

        if (it != end && *it != '}') {
            throw format_error("Invalid format specifier for Eigen matrix");
        }

        return it;
    }

    template <typename FormatContext>
    auto format(const Eigen::MatrixBase<derived> &mat, FormatContext &ctx) const -> FormatContext::iterator {
        std::ostringstream oss;
        if (precision >= 0) {
            Eigen::IOFormat cleanFmt(precision, 0, ", ", "\n", "[", "]");
            oss << mat.format(cleanFmt);
        } else {
            oss << mat;
        }
        return fmt::format_to(ctx.out(), "{}", oss.str());
    }
};
} // namespace fmt

#endif
