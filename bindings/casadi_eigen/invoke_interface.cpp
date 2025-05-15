#include <atri/core/fwd.hpp>
#include <functional>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>

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

namespace py = pybind11;

/// @todo do I need to check major order?
using vector = Eigen::VectorXd;
using matrix = Eigen::MatrixXd;

template <typename in_t, typename out_t>
using func_ptr = void (*)(std::vector<Eigen::Ref<in_t>> &,
                          std::vector<Eigen::Ref<out_t>> &);

using func_vec_mat_ptr = func_ptr<vector, matrix>;
using func_vec_vec_ptr = func_ptr<vector, vector>;

template <typename in_t, typename out_t>
py::capsule load_func(const std::string &lib_path, const std::string &func_name) {
    void *handle = LOAD_LIBRARY(lib_path.c_str());
    if (!handle)
        throw std::runtime_error("Failed to open library");
    void *sym = GET_SYMBOL(handle, func_name.c_str());
    if (!sym)
        throw std::runtime_error("Failed to get function symbol");
    auto func = reinterpret_cast<func_ptr<in_t, out_t>>(sym);
    return py::capsule((void *)func, "func_ptr");
}

template <typename in_t, typename out_t>
inline void dispatch_function(
    // func_ptr<in_t, out_t> func,
    py::capsule func,
    py::list in,
    py::list out) {

    std::vector<Eigen::Ref<in_t>> input_refs;
    std::vector<Eigen::Ref<out_t>> output_refs;

    for (py::handle m : in)
        input_refs.emplace_back(m.cast<Eigen::Ref<in_t>>());
    for (py::handle m : out)
        output_refs.emplace_back(m.cast<Eigen::Ref<out_t>>());

    auto f = reinterpret_cast<func_ptr<in_t, out_t>>(func.get_pointer());

    f(input_refs, output_refs);
}

PYBIND11_MODULE(invoke_interface, m) {
    m.def("load_vec_vec", &load_func<vector, vector>,
          py::arg("lib_path"),
          py::arg("func_name"));
    m.def("load_vec_mat", &load_func<vector, matrix>,
          py::arg("lib_path"),
          py::arg("func_name"));
    m.def("invoke_vec_vec", &dispatch_function<vector, vector>,
          py::arg("func"),
          py::arg("in"),
          py::arg("out"));
    m.def("invoke_vec_mat", &dispatch_function<vector, matrix>,
          py::arg("func"),
          py::arg("in"),
          py::arg("out"));
}