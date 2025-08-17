#ifndef MOTO_UTILS_FUNC_TRAITS_HPP
#define MOTO_UTILS_FUNC_TRAITS_HPP

#include <functional>

namespace moto {
namespace utils {
template <typename T>
struct func_traits {
    static constexpr size_t arg_num = size_t(-1); ///< number of arguments
};

// Primary template for function types R(Args...)
template <typename R, typename... Args>
struct func_traits<R(Args...)> {
    using return_type = R;
    using arg_types = std::tuple<Args...>;
    static constexpr size_t arg_num = sizeof...(Args);
    template <size_t i>
    using arg_type = std::tuple_element_t<i, arg_types>;
};

// Function pointers R(*)(Args...)
template <typename R, typename... Args>
struct func_traits<R (*)(Args...)> : func_traits<R(Args...)> {
    using function_pointer_type = R (*)(Args...);
};

// Function references R(&)(Args...)
template <typename R, typename... Args>
struct func_traits<R (&)(Args...)> : func_traits<R(Args...)> {
    using function_reference_type = R (&)(Args...);
};

// std::function<R(Args...)>
template <typename R, typename... Args>
struct func_traits<std::function<R(Args...)>> : func_traits<R(Args...)> {
    using function_type = std::function<R(Args...)>;
};

// Non-const member functions
template <typename Class, typename R, typename... Args>
struct func_traits<R (Class::*)(Args...)> : func_traits<R(Args...)> {
    using member_function_type = R (Class::*)(Args...);
    using class_type = Class;
};

// Const member functions
template <typename Class, typename R, typename... Args>
struct func_traits<R (Class::*)(Args...) const> : func_traits<R(Args...)> {
    using member_function_type = R (Class::*)(Args...) const;
    using class_type = Class;
};

// Volatile member functions
template <typename Class, typename R, typename... Args>
struct func_traits<R (Class::*)(Args...) volatile> : func_traits<R(Args...)> {
    using member_function_type = R (Class::*)(Args...) volatile;
    using class_type = Class;
};

// Const volatile member functions
template <typename Class, typename R, typename... Args>
struct func_traits<R (Class::*)(Args...) const volatile> : func_traits<R(Args...)> {
    using member_function_type = R (Class::*)(Args...) const volatile;
    using class_type = Class;
};

// Lvalue ref-qualified member functions
template <typename Class, typename R, typename... Args>
struct func_traits<R (Class::*)(Args...) &> : func_traits<R(Args...)> {
    using member_function_type = R (Class::*)(Args...) &;
    using class_type = Class;
};

template <typename Class, typename R, typename... Args>
struct func_traits<R (Class::*)(Args...) const &> : func_traits<R(Args...)> {
    using member_function_type = R (Class::*)(Args...) const &;
    using class_type = Class;
};

template <typename Class, typename R, typename... Args>
struct func_traits<R (Class::*)(Args...) volatile &> : func_traits<R(Args...)> {
    using member_function_type = R (Class::*)(Args...) volatile &;
    using class_type = Class;
};

template <typename Class, typename R, typename... Args>
struct func_traits<R (Class::*)(Args...) const volatile &> : func_traits<R(Args...)> {
    using member_function_type = R (Class::*)(Args...) const volatile &;
    using class_type = Class;
};

// Rvalue ref-qualified member functions
template <typename Class, typename R, typename... Args>
struct func_traits<R (Class::*)(Args...) &&> : func_traits<R(Args...)> {
    using member_function_type = R (Class::*)(Args...) &&;
    using class_type = Class;
};

template <typename Class, typename R, typename... Args>
struct func_traits<R (Class::*)(Args...) const &&> : func_traits<R(Args...)> {
    using member_function_type = R (Class::*)(Args...) const &&;
    using class_type = Class;
};

template <typename Class, typename R, typename... Args>
struct func_traits<R (Class::*)(Args...) volatile &&> : func_traits<R(Args...)> {
    using member_function_type = R (Class::*)(Args...) volatile &&;
    using class_type = Class;
};

template <typename Class, typename R, typename... Args>
struct func_traits<R (Class::*)(Args...) const volatile &&> : func_traits<R(Args...)> {
    using member_function_type = R (Class::*)(Args...) const volatile &&;
    using class_type = Class;
};

// Noexcept versions (C++17 feature but commonly used with C++20)
template <typename R, typename... Args>
struct func_traits<R (*)(Args...) noexcept> : func_traits<R(Args...)> {
    using function_pointer_type = R (*)(Args...) noexcept;
    static constexpr bool is_noexcept = true;
};

template <typename Class, typename R, typename... Args>
struct func_traits<R (Class::*)(Args...) noexcept> : func_traits<R(Args...)> {
    using member_function_type = R (Class::*)(Args...) noexcept;
    using class_type = Class;
    static constexpr bool is_noexcept = true;
};

template <typename Class, typename R, typename... Args>
struct func_traits<R (Class::*)(Args...) const noexcept> : func_traits<R(Args...)> {
    using member_function_type = R (Class::*)(Args...) const noexcept;
    using class_type = Class;
    static constexpr bool is_noexcept = true;
};

// C++20 concept to check if a type is callable
template <typename T>
concept Callable = requires(T t) {
    &std::decay_t<T>::operator();
};

// Lambda and callable objects - using C++20 concepts for better SFINAE
template <Callable T>
    requires(!std::is_function_v<T> &&
             !std::is_member_function_pointer_v<T> &&
             !std::is_function_v<std::remove_pointer_t<T>>)
struct func_traits<T> : func_traits<decltype(&std::decay_t<T>::operator())> {};
} // namespace utils
} // namespace moto

#endif