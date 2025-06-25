#include <moto/core/fwd.hpp>
#include <functional>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
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

struct external_function {
    external_function(const external_function& rhs) = default;
    explicit external_function(const std::string &lib_path, const std::string &func_name) {
        void *handle = LOAD_LIBRARY(lib_path.c_str());
        if (!handle)
            throw std::runtime_error("Failed to open library");
        void *sym = GET_SYMBOL(handle, func_name.c_str());
        if (!sym)
            throw std::runtime_error("Failed to get function symbol");
        func_ptr = py::capsule(sym, "func_ptr");
    }

    void operator()(py::list in, py::list out) {
        using in_t = moto::vector;
        using out_t = moto::matrix;

        std::vector<Eigen::Ref<in_t>> input_refs;
        std::vector<Eigen::Ref<out_t>> output_refs;

        for (py::handle m : in)
            input_refs.emplace_back(m.cast<Eigen::Ref<in_t>>());
        for (py::handle m : out) {
            py::array_t<double, py::array::c_style | py::array::forcecast> arr = py::cast<py::array>(m);
            if (arr.ndim() != 2)
                throw std::runtime_error("Each output must be 2D");
            output_refs.emplace_back(Eigen::Map<out_t>(
                arr.mutable_data(), arr.shape(0), arr.shape(1)));
        }

        auto f = reinterpret_cast<void (*)(decltype(input_refs) &, decltype(output_refs) &)>(func_ptr.get_pointer());
        f(input_refs, output_refs);
    }

    void operator()(py::list in, Eigen::Ref<moto::vector> out) {
        using in_t = moto::vector;
        using out_t = moto::vector;

        std::vector<Eigen::Ref<in_t>> input_refs;

        for (py::handle m : in)
            input_refs.emplace_back(m.cast<Eigen::Ref<in_t>>());

        auto f = reinterpret_cast<void (*)(decltype(input_refs) &, decltype(out))>(func_ptr.get_pointer());
        f(input_refs, out);
    }

  private:
    py::capsule func_ptr;
};

void register_submodule_codegen(pybind11::module_ &m) {
    py::class_<external_function>(m, "external_function")
        .def(py::init<const std::string &, const std::string &>(), py::arg("lib_path"), py::arg("func_name"))
        .def("__call__", static_cast<void (external_function::*)(py::list, py::list)>(&external_function::operator()),
             py::arg("in"),
             py::arg("out"))
        .def("__call__", static_cast<void (external_function::*)(py::list, Eigen::Ref<moto::vector>)>(&external_function::operator()),
             py::arg("in"),
             py::arg("out"));
}
