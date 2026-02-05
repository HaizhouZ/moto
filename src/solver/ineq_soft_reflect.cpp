#include <moto/ocp/ineq_constr.hpp>
#include <moto/solver/ipm/ipm_constr.hpp>

namespace moto {
namespace details {
/// @brief bind constructor for inequality constraint derived types
/// @tparam T  derived type
/// @return constructor function
template <typename T>
    requires std::is_base_of_v<generic_constr, T> && std::is_constructible_v<T, generic_constr &&>
auto ineq_bind_constructor() {
    return std::function<generic_constr *(generic_constr &)>([](generic_constr &self) {
        return static_cast<generic_constr *>(new T(std::move(static_cast<generic_constr &>(self))));
    });
}
/// @brief registry of inequality constraint derived types
#define ADD_INEQ_REGISTRY_ENTRY(T) \
    {#T, ineq_bind_constructor<T>()}
/// @brief map of inequality constraint derived types
std::map<std::string, decltype(ineq_bind_constructor<generic_constr>()), std::less<>> ineq_derived_registry = {
    ADD_INEQ_REGISTRY_ENTRY(ipm),
};
} // namespace details

generic_constr *generic_constr::as_ineq(std::string_view type_name) {
    auto it = detials::ineq_derived_registry.find(type_name);
    if (it != detials::ineq_derived_registry.end()) {
        return it->second(*this);
    } else
        throw std::runtime_error("Unknown inequality constraint type: " + std::string(type_name));
}
} // namespace moto