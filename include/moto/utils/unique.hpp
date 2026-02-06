#ifndef MOTO_UTILS_UNIQUE_HPP
#define MOTO_UTILS_UNIQUE_HPP

#include <moto/utils/clone_traits.hpp>
#include <moto/utils/func_traits.hpp>
namespace moto {
namespace utils {

/**
 * @brief a unique pointer wrapper that supports cloning for copy constructor and copy assignment operator
 *
 * @tparam T the type of the object being pointed to, must be clonable for copy constructor and copy assignment operator to work
 * @tparam clone if true, clone() will be called; otherwise the default constructor will be called
 */
template <typename T, bool clone = true>
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
        : std::unique_ptr<T>(static_cast<T *>(other ? other->clone()
                                                    : nullptr)) {
        static_assert(clone, "Cloning is disabled for this unique pointer");
    }

    /// @brief copy constructor
    /// @param other
    unique(const unique &other) noexcept
        requires(clone)
        : std::unique_ptr<T>(static_cast<T *>(other ? other->clone()
                                                    : nullptr)) {
        static_assert(is_clonable<T>, "Type T must be clonable to use copy constructor");
    }
    /// @brief copy constructor when cloning is disabled, will call default constructor of T
    unique(const unique &other) noexcept
        requires(!clone)
        : std::unique_ptr<T>(new T()) {}

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

    unique()
        requires(std::is_default_constructible_v<T>)
        : base(std::make_unique<T>()) {} ///< default constructor

    /// @brief bool conversion operator, true if the pointer is not null
    operator bool() const noexcept {
        return this->get() != nullptr;
    }
    /// @brief equality operator by comparing the underlying objects
    friend bool operator==(const unique &lhs, const T &rhs) noexcept {
        return *(lhs.get()) == rhs;
    }

    /// @brief replace the underlying object by applying a callback function to the current object, and return a reference to the new object
    /// @tparam U the type of the new object, must be convertible to T
    /// @tparam callback_t the type of the callback function, must be callable with a T& argument and return a pointer
    /// @param callback
    /// @return
    template <callable callback_t>
    decltype(auto) replace(callback_t callback) {
        using func_type = func_traits<callback_t>;
        using return_type = typename func_type::return_type;
        static_assert(std::is_pointer_v<return_type>, 
                      "Callback function must return a pointer type");
        static_assert(std::is_convertible_v<return_type, T*>,
                      "Callback return type must be convertible to T*");
        
        this->reset(static_cast<T*>(callback(*this)));
        return (std::remove_pointer_t<return_type>&)(*this);
    }
};
} // namespace utils
} // namespace moto

#endif