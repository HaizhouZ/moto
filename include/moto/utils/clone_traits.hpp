#ifndef MOTO_UTILS_CLONE_TRAIT_HPP
#define MOTO_UTILS_CLONE_TRAIT_HPP
#include <concepts>

namespace moto {
namespace utils {
/**
 * @brief clone_base interface for objects that can be cloned
 *
 * @tparam T the base type of the clone_base object, used for covariant return type in clone()
 */
template <typename T>
struct clone_base {
    clone_base() = default;
    using clone_ptr = clone_base *;
    virtual clone_ptr clone() const = 0; ///< clone the object
    virtual ~clone_base() = default;
};

#define DEF_DEFAULT_CLONE(cls) \
    clone_ptr clone() const override { return new cls(*this); } ///< default clone implementation

/**
 * @brief lineage concept to check if two types are in the same
 * inheritance hierarchy (i.e., one is derived from the other or they are the same)
 */
template <typename T, typename U>
concept lineaged =
    std::is_base_of_v<std::remove_cvref_t<T>, std::remove_cvref_t<U>> ||
    std::is_base_of_v<std::remove_cvref_t<U>, std::remove_cvref_t<T>> ||
    std::is_same_v<T, U>;

/// @brief concept to check if a type is clonable (i.e., derived from clone_base)
template <typename T>
concept is_clonable = requires(std::unwrap_ref_decay_t<T> a) {
    { a.clone() } -> std::same_as<typename std::unwrap_ref_decay_t<T>::clone_ptr>;
};
} // namespace utils
} // namespace moto
#endif /* MOTO_UTILS_CLONE_TRAITS_HPP */