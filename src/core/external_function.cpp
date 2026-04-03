#include <moto/core/external_function.hpp>

#include <chrono>
#include <filesystem>
#include <thread>

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

namespace moto {
void *load_from_shared(const std::string &lib_path, const std::string &func_name) {
    std::string last_error;
    for (int attempt = 0; attempt < 20; ++attempt) {
#ifndef _WIN32
        dlerror();
#endif
        void *handle = LOAD_LIBRARY(lib_path.data());
        if (handle) {
#ifndef _WIN32
            dlerror();
#endif
            void *func_sym = GET_SYMBOL(handle, func_name.data());
            if (func_sym) {
                return func_sym;
            }
#ifdef _WIN32
            last_error = fmt::format("GetProcAddress failed for {}", func_name);
#else
            if (const char *err = dlerror()) {
                last_error = err;
            } else {
                last_error = fmt::format("missing symbol {}", func_name);
            }
#endif
            CLOSE_LIBRARY(handle);
        } else {
#ifdef _WIN32
            last_error = fmt::format("LoadLibrary failed for {}", lib_path);
#else
            if (const char *err = dlerror()) {
                last_error = err;
            } else {
                last_error = "unknown dlopen error";
            }
#endif
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    throw std::runtime_error(fmt::format("Failed to load symbol {} from {}: {}", func_name, lib_path, last_error));
}

ext_func::ext_func(const std::string &func_name, const std::string &lib_path) {
    std::filesystem::path p(lib_path);
    func_ = load_from_shared(p / ("lib" + func_name + ".so"), func_name);
}

std::array<ext_func, 3> load_approx(const std::string &name,
                                    bool load_eval, bool load_jac, bool load_hess,
                                    const std::string &path) {
    std::array<ext_func, 3> funcs;
    if (load_eval)
        funcs[0] = ext_func(name, path);
    if (load_jac)
        funcs[1] = ext_func(name + "_jac", path);
    if (load_hess)
        funcs[2] = ext_func(name + "_hess", path);
    return funcs;
}

} // namespace moto
