#include <moto/core/fields.hpp>
#include <binding_fwd.hpp>

using namespace moto;
namespace nb = nanobind;

// Template function to export any enum using magic_enum
void export_field_with_magic(nb::module_ &m, const std::string &python_name) {
    nb::enum_<field_t> enum_binder(m, python_name.c_str());

    // Iterate over all enum values provided by magic_enum
    for (auto [value, name] : magic_enum::enum_entries<field_t>()) {
        if (value != field_t::NUM)
            enum_binder.value(("field_" + std::string(name).substr(2)).c_str(), value);
        else
            enum_binder.value("field_NUM", value);
    }
    enum_binder.export_values(); // Makes enum members accessible like MyEnum.MEMBER
}

void register_submodule_fields(nb::module_ &m) {
    export_field_with_magic(m, "field");
}