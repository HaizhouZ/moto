#include <atri/core/external_function.hpp>

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
void *load_from_shared(const std::string &lib_path, const std::string &func_name) {
    void *handle = LOAD_LIBRARY(lib_path.data());
    if (!handle)
        throw std::runtime_error(fmt::format("Failed to open library: {}", lib_path));
    void *func_sym = GET_SYMBOL(handle, func_name.data());
    if (!func_sym)
        throw std::runtime_error(fmt::format("Failed to get function symbol: {}", func_name));
    return func_sym;
}

std::array<ext_func, 3> load_approx(const std::string &name,
                                    bool load_eval, bool load_jac, bool load_hess,
                                    const std::string &path) {
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