#ifndef ATRI_CORE_ARRAY_HPP
#define ATRI_CORE_ARRAY_HPP

#include <cassert>
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
struct shifted_array : public std::array<T, N> {
    using base_type = std::array<T, N>;
    auto &operator[](size_t i) {
        assert(i >= st && i < st + N);
        return base_type::operator[](i - st);
    }
    const auto &operator[](size_t i) const {
        assert(i >= st && i < st + N);
        return base_type::operator[](i - st);
    }
    shifted_array() = default;
    shifted_array(base_type &&rhs) : base_type(std::move(rhs)) {}
};

template <typename T, size_t N>
using array = shifted_array<T, N, 0>;

} // namespace atri

#endif // ATRI_CORE_ARRAY_HPP