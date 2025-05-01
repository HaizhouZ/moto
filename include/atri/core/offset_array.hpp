#ifndef ATRI_CORE_OFFSET_ARRAY_HPP
#define ATRI_CORE_OFFSET_ARRAY_HPP

#include <stdexcept>
#include <vector>

namespace atri {
/**
 * @brief array with offset for indexing
 *
 * @tparam T type to store in array
 * @tparam N number elements in array
 * @tparam st offset or starting idx value (the [st] correspond to val[0])
 */
template <typename T, size_t N, size_t st>
struct offset_array {
    std::array<T, N> val;
    auto &operator[](size_t i) {
        return val[i - st];
    }
    const auto &operator[](size_t i) const {
        return val[i - st];
    }
    offset_array() = default;
    offset_array(offset_array<T, N, st> &&rhs) : val(std::move(rhs.val)) {}
};
} // namespace atri

#endif // ATRI_CORE_OFFSET_ARRAY_HPP