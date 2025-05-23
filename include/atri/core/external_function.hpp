#ifndef ATRI_EVAL_EXTERNAL_FUNCTION_HPP
#define ATRI_EVAL_EXTERNAL_FUNCTION_HPP

#include <atri/core/fwd.hpp>
#include <filesystem>

#ifdef _WIN32
#include <windows.h>
#define LIB_HANDLE HMODULE
#define LOAD_LIBRARY(name) LoadLibraryA(name)
#define GET_SYMBOL(handle, symbol) GetProcAddress(handle, symbol)
#define CLOSE_LIBRARY(handle) FreeLibrary(handle)
#else
#include <dlfcn.h>
#define LIB_HANDLE void *
#define LOAD_LIBRARY(name) dlopen(name, RTLD_NOW)
#define GET_SYMBOL(handle, symbol) dlsym(handle, symbol)
#define CLOSE_LIBRARY(handle) dlclose(handle)
#endif

namespace atri {
template <typename in_t, typename out_t>
using func_ptr = void (*)(std::vector<Eigen::Ref<in_t>> &,
                          std::vector<Eigen::Ref<out_t>> &);

inline void *__load_from_shared(const std::string &lib_path, const std::string &func_name) {
    void *handle = LOAD_LIBRARY(lib_path.data());
    if (!handle)
        throw std::runtime_error(fmt::format("Failed to open library: {}", lib_path));
    void *func_sym = GET_SYMBOL(handle, func_name.data());
    if (!func_sym)
        throw std::runtime_error(fmt::format("Failed to get function symbol: {}", func_name));
    return func_sym;
}

template <typename in_t, typename out_t>
inline auto load_from_shared(const std::string &lib_path, const std::string &func_name) {
    return reinterpret_cast<func_ptr<in_t, out_t>>(__load_from_shared(lib_path, func_name));
}

struct ext_func {
    void *func_;
    ext_func() : func_(nullptr) {}
    ext_func(const std::string &lib_path, const std::string &func_name)
        : func_(__load_from_shared(lib_path, func_name)) {}

    template <typename in_t, typename out_t>
    auto invoke(in_t &input,
                out_t &output) const {
        return reinterpret_cast<void (*)(decltype(input), decltype(output))>(func_)(input, output);
    }
};
inline auto load_approx(const std::string &name,
                 bool load_eval, bool load_jac, bool load_hess,
                 const std::string &path = "gen") {
    std::array<ext_func, 3> funcs;
    std::filesystem::path p(path);
    if (load_eval)
        funcs[0] = ext_func(p / ("lib" + name + ".so"), name);
    if (load_jac)
        funcs[1] = ext_func(p / ("lib" + name + "_jac.so"), name + "_jac");
    if (load_hess)
        funcs[2] = ext_func(p / ("lib" + name + "_hess.so"), name + "_hess");
    return funcs;
}
} // namespace atri

#endif // ATRI_EVAL_EXTERNAL_FUNCTION_HPP