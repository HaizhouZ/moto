#include <moto/ocp/constr.hpp>
#include <moto/solver/soft_constr/pmm_constr.hpp>

namespace moto {
namespace details {
/// @brief bind constructor for soft equality constraint derived types
template <typename T>
    requires std::is_base_of_v<soft_constr, T> && std::is_constructible_v<T, generic_constr &&>
auto soft_bind_constructor() {
    return std::function<generic_constr *(generic_constr &)>([](generic_constr &self) {
        return static_cast<generic_constr *>(new T(std::move(static_cast<generic_constr &>(self))));
    });
}
#define ADD_SOFT_REGISTRY_ENTRY(T) \
    {#T, soft_bind_constructor<T>()}
std::map<std::string, decltype(soft_bind_constructor<pmm_constr>()), std::less<>> soft_derived_registry = {
    ADD_SOFT_REGISTRY_ENTRY(pmm_constr),
};
} // namespace details

generic_constr *generic_constr::cast_soft(std::string_view type_name) {
    auto it = details::soft_derived_registry.find(type_name);
    if (it != details::soft_derived_registry.end()) {
        return it->second(*this);
    } else
        throw std::runtime_error("Unknown soft constraint type: " + std::string(type_name));
}
} // namespace moto
