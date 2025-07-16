#ifndef MOTO_CORE_ARRAY_HPP
#define MOTO_CORE_ARRAY_HPP

#include <cassert>
#include <stdexcept>
#include <vector>

namespace moto {
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
    using base_type::base_type; // inherit constructors
};

template <typename T, size_t N>
using array = shifted_array<T, N, 0>;


template <typename T, std::array arr>
    requires std::is_constructible_v<size_t, decltype(arr[0])>
using array_type = shifted_array<T, std::tuple_size_v<decltype(arr)>, arr[0]>;

} // namespace moto

#endif // MOTO_CORE_ARRAY_HPP