#include <moto/core/fields.hpp>
#include <binding_fwd.hpp>
#include <enum_export.hpp>
namespace nb = nanobind;

void register_submodule_fields(nb::module_ &m) {
    moto::export_enum<moto::field_t>(
        m,
        "Internal storage and solver role assigned to symbols and expressions. "
        "Ordinary modeling APIs infer this automatically.");
}
