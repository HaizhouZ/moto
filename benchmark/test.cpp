#include <functional>
#include <memory>

/// @brief trait to check if a type is a std::shared_ptr
template <typename>
struct is_shared_ptr_ : std::false_type {};
/// specialization for std::shared_ptr
template <typename T>
struct is_shared_ptr_<std::shared_ptr<T>> : std::true_type {};

/// @brief Helper concept using the trait (handling references/const)
template <typename T>
concept is_shared_ptr = is_shared_ptr_<std::remove_cvref_t<T>>::value;

/// @brief concept to check if a type is shareable (i.e., derived from enable_shared_from_this)
template <typename T>
concept shareable = requires(std::unwrap_ref_decay_t<T> a) {
    { a.shared_from_this() } -> is_shared_ptr;
};

template <shareable T>
struct shared_type {
    using value = decltype(std::declval<std::unwrap_ref_decay_t<T>>().shared_from_this())::element_type;
};
struct A : std::enable_shared_from_this<A> {};
shared_type<const std::reference_wrapper<A>&>::value a;