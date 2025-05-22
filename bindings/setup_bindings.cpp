#include <pybind11/pybind11.h>

extern void register_submodule_codegen(pybind11::module_ &m);
extern void register_submodule_fields(pybind11::module_ &m);

PYBIND11_MODULE(atri_pywrap, m) {
    register_submodule_codegen(m);
    register_submodule_fields(m);
}