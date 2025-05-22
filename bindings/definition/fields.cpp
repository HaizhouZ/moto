#include <atri/core/fields.hpp>
#include <pybind11/pybind11.h>

using namespace atri;
namespace py = pybind11;

// Template function to export any enum using magic_enum
void export_enum_with_magic(py::module_ &m, const std::string &python_name) {
    py::enum_<field_t> enum_binder(m, python_name.c_str());

    // Iterate over all enum values provided by magic_enum
    for (auto [value, name] : magic_enum::enum_entries<field_t>()) {
        if (value != field_t::NUM)
            enum_binder.value(("field_" + std::string(name).substr(2)).c_str(), value);
    }
    enum_binder.export_values(); // Makes enum members accessible like MyEnum.MEMBER
}

void register_submodule_fields(pybind11::module_ &m) {
    export_enum_with_magic(m, "field");
}