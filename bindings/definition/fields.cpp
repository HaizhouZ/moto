#include <moto/core/fields.hpp>
#include <binding_fwd.hpp>
#include <enum_export.hpp>
namespace nb = nanobind;

void register_submodule_fields(nb::module_ &m) {
    moto::export_enum<moto::field_t>(m);
}