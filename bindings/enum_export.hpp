#include <moto/core/fwd.hpp>

#include <nanobind/nanobind.h>

namespace moto {

template <typename T>
void export_enum(nb::handle &m) {
    std::string enum_type_name{magic_enum::enum_type_name<T>()};
    // remove _t suffix if exists
    if (enum_type_name.size() > 2 && enum_type_name.substr(enum_type_name.size() - 2) == "_t") {
        enum_type_name = enum_type_name.substr(0, enum_type_name.size() - 2);
    }
    nb::enum_<T> enum_binder(m, enum_type_name.c_str());

    // Iterate over all enum values provided by magic_enum
    for (auto [value, name] : magic_enum::enum_entries<T>()) {
        enum_binder.value((fmt::format("{}_{}", enum_type_name, name)).c_str(), value);
    }
    enum_binder.export_values(); // Makes enum members accessible like MyEnum.MEMBER
}

} // namespace moto