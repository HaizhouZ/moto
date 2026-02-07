#ifndef MOTO_UTILS_SHARED_HPP
#define MOTO_UTILS_SHARED_HPP
#include <functional>
#include <memory>
#include <moto/utils/clone_traits.hpp>

namespace moto {
namespace utils {

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

/// @brief trait to extract the value type of a shareable type
template <shareable T>
struct shared_type {
    using value = decltype(std::declval<std::unwrap_ref_decay_t<T>>().shared_from_this())::element_type;
};

template <typename T>
class shared : public std::shared_ptr<T> {
  private:
    template <typename U>
    friend class shared;

  public:
    /// @copy constructors from std::shared_ptr
    /// @warning: make sure other points to a convertible class object to T
    /// @details: delegate to moving constructor with a copy of other
    template <typename U>
    shared(const std::shared_ptr<U> &other) : shared(std::shared_ptr<U>(other)) {}
    /// @brief move constructor from std::shared_ptr<U>
    /// @warning: make sure other points to a convertible class object to T
    template <typename U>
    shared(std::shared_ptr<U> &&other) : std::shared_ptr<T>(std::static_pointer_cast<T>(other)) {}

    /// @brief constructor from raw pointer U* and take the ownership
    template <typename U>
    shared(U *ptr) : shared(std::shared_ptr<U>(ptr)) {}

    /// @brief move constructor from shared<U>
    /// @warning: make sure other points to a convertible class object to T
    /// @details: after moving, other will be invalidated
    template <typename U>
    shared(shared<U> &&other) noexcept
        : std::shared_ptr<T>(std::static_pointer_cast<T>(std::move(other))) {
    }
    /// @brief copy constructor from shared<U>
    /// @warning: make sure other points to a convertible class object to T
    /// @details: after copying, both this and other will point to the same object
    template <typename U>
    shared(const shared<U> &other) noexcept
        : std::shared_ptr<T>(std::static_pointer_cast<T>(other)) {
    }

    /// @brief constructor by value @warning: U must be derived from enable_shared_from_this<...>
    /// @note it will remove the const qualifier from U if any
    /// @warning rhs must be held by a shared_ptr, otherwise it will throw std::bad_weak_ptr
    template <shareable U, typename value_type = typename shared_type<U>::value>
    shared(U &&rhs)
        : shared(
              /// 2. Downcast Base -> Derived
              std::static_pointer_cast<T>(
                  /// 1. Remove const from Base
                  std::const_pointer_cast<value_type>(
                      /// 0. deference to call shared_from_this()
                      ((const value_type &)rhs).shared_from_this()))) {}

    shared() noexcept = default; ///< default constructor

    void swap(shared &other) noexcept {
        std::shared_ptr<T>::swap(other);
    }

    template <lineaged<T> U>
    shared &operator=(const shared<U> &other) noexcept {
        shared(other).swap(*this);
        return *this;
    }

    template <lineaged<T> U>
    shared &operator=(shared<U> &&other) noexcept {
        shared(std::move(other)).swap(*this);
        return *this;
    }

    /// @brief reference operator to U
    /// @warning: make sure this points to a convertible class object to U
    template <lineaged<T> U>
    operator U &() const noexcept {
        return *static_cast<U *>(this->get());
    }

    using std::shared_ptr<T>::operator->; ///< inherit operator-> from std::shared_ptr

    /// @brief conversion to U&, alias of operator U&()
    template <lineaged<T> U>
    U &as() const noexcept {
        return this->operator U &();
    }

    /// @brief conversion to shared<U>
    template <lineaged<T> U>
    shared<U> cast() const noexcept {
        return shared<U>(*this);
    }

    /// @brief conversion to bool operator @return true if the pointer is not null
    operator bool() const noexcept {
        return this->get() != nullptr;
    }

    bool operator!() const noexcept {
        return !this->operator bool();
    }

    /// @brief clone the object pointed to @return shared<T> to the cloned object
    shared<T> clone() const {
        static_assert(is_clonable<T>, "Type T must be clonable to use clone()");
        if (!bool(*this)) {
            throw std::runtime_error("Cannot clone null object");
        }
        return shared<T>(this->get()->clone());
    }
    /// @brief equality operator by comparing the underlying objects
    friend bool operator==(const shared &lhs, const T &rhs) noexcept {
        return lhs && *(lhs.get()) == rhs;
    }
};
} // namespace utils
template <typename T>
inline std::string format_as(const utils::shared<T> &e) {
    return format_as(*e);
}
} // namespace moto
#endif // MOTO_UTILS_SHARED_HPP