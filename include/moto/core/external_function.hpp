#ifndef MOTO_EVAL_EXTERNAL_FUNCTION_HPP
#define MOTO_EVAL_EXTERNAL_FUNCTION_HPP

#include <moto/core/fwd.hpp>

namespace moto {
/**
 * @brief Load a function from a shared library
 * This function loads a symbol from a shared library specified by `lib_path`.
 * It uses platform-specific APIs to load the library and retrieve the function pointer.
 * @param lib_path Path to the shared library (e.g., "gen").
 * @param func_name Name of the function to load (e.g., "my_approx").
 * @return Pointer to the loaded function
 * @exception Throws std::runtime_error if the library cannot be opened or the function cannot be found.
 */
void *load_from_shared(const std::string &lib_path, const std::string &func_name);
/**
 * @brief A wrapper for an external function loaded from a shared library
 * This struct holds a pointer to the loaded function and provides an interface to invoke it.
 */
struct ext_func {
    void *func_;
    ext_func() : func_(nullptr) {}
    /**
     * @brief Check if the function pointer is empty
     * @return true if the function pointer is null, false otherwise
     */
    bool empty() { return func_ == nullptr; }
    /**
     * @brief Construct a new ext_func object by calling @ref load_from_shared
     * @param lib_path Path to the shared library (e.g., "gen/libmy_approx.so")
     * @param func_name Name of the function to load (e.g., "my_approx")
     */
    ext_func(const std::string &lib_path, const std::string &func_name)
        : func_(load_from_shared(lib_path, func_name)) {}
    /**
     * @brief Invoke the loaded function
     *
     * @tparam in_t input type
     * @tparam out_t output type
     * @param input input arg, e.g. std::vector<vector::ref>
     * @param output output arg, e.g. std::vector<vector::ref>
     */
    template <typename in_t, typename out_t>
    void invoke(in_t &input,
                out_t &output) const {
        reinterpret_cast<void (*)(decltype(input), decltype(output))>(func_)(input, output);
    }
};
/**
 * @brief Load an external approximation function
 * automatically loads the function from a shared lib of name libname.so in the specified path
 * @param name name of the function, e.g., "my_approx"
 * @param load_eval true to load the evaluation function of the name
 * @param load_jac  true to load the jacobian function of the name + _jac
 * @param load_hess true to load the hessian function of the name + _hess
 * @param path path to the shared library, default is "gen"
 * @return std::array<ext_func, 3> loaded functions, (eval, jac, hess)
 */
std::array<ext_func, 3> load_approx(const std::string &name,
                                    bool load_eval, bool load_jac, bool load_hess,
                                    const std::string &path = "gen");
} // namespace moto

#endif // MOTO_EVAL_EXTERNAL_FUNCTION_HPP