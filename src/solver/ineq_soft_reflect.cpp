#include <moto/ocp/ineq_constr.hpp>
#include <moto/solver/ipm/ipm_constr.hpp>
#include <moto/solver/soft_constr/pmm_constr.hpp>

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

generic_constr *generic_constr::cast_ineq(std::string_view type_name) {
    auto it = details::ineq_derived_registry.find(type_name);
    if (it != details::ineq_derived_registry.end()) {
        return it->second(*this);
    } else
        throw std::runtime_error("Unknown inequality constraint type: " + std::string(type_name));
}

generic_constr *generic_constr::cast_soft(std::string_view type_name) {
    auto it = details::soft_derived_registry.find(type_name);
    if (it != details::soft_derived_registry.end()) {
        return it->second(*this);
    } else
        throw std::runtime_error("Unknown soft constraint type: " + std::string(type_name));
}
} // namespace moto