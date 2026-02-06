#ifndef MOTO_UTILS_UNIQUE_HPP
#define MOTO_UTILS_UNIQUE_HPP

#include <moto/utils/clone_traits.hpp>

namespace moto {
namespace utils {

template <typename T>
class unique : public std::unique_ptr<T> {

  public:
    using base = std::unique_ptr<T>;
    using base::base; ///< inherit constructors from std::unique_ptr

    /// @brief move constructor from unique<U>
    /// @warning: make sure other points to a convertible class object to T
    template <typename U>
    unique(unique<U> &&other) noexcept
        : unique(other.release()) {}

    /// @brief copy constructor from unique<U>
    /// @param other
    template <is_clonable U>
    unique(const unique<U> &other) noexcept
        : std::unique_ptr<T>(static_cast<T *>(other ? other->clone() : nullptr)) {
    }

    /// @brief copy constructor
    /// @param other
    unique(const unique &other) noexcept
        : std::unique_ptr<T>(static_cast<T *>(other ? other->clone() : nullptr)) {
        static_assert(is_clonable<T>, "Type T must be clonable to use copy constructor");
    }

    /// @brief move constructor from std::unique_ptr<U>
    /// @param other
    template <typename U>
    unique(std::unique_ptr<U> &&other) noexcept
        : std::unique_ptr<T>(static_cast<T *>(other.release())) {}

    /// @brief reference operator to U
    /// @warning: make sure this points to a convertible class object to U
    template <typename U>
    operator U &() const noexcept {
        return *static_cast<U *>(this->get());
    }

    /// @brief bool conversion operator, true if the pointer is not null
    operator bool() const noexcept {
        return this->get() != nullptr;
    }
    /// @brief equality operator by comparing the underlying objects
    friend bool operator==(const unique &lhs, const T &rhs) noexcept {
        return *(lhs.get()) == rhs;
    }
};
} // namespace utils
} // namespace moto

#endif