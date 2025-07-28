#include <moto/ocp/ineq_constr.hpp>
#include <moto/solver/ipm/ipm_constr.hpp>

namespace moto {
std::map<std::string, const std::type_info *, std::less<>> ineq_derived_registry = {
    {"ipm", &typeid(ipm)}};

constr& constr::as_ineq(std::string_view type_name) {
    auto it = ineq_derived_registry.find(type_name);
    if (it != ineq_derived_registry.end()) {
        const std::type_info *type_info = it->second;
        if (*type_info == typeid(ipm)) {
            return as_ineq<ipm>();
        }
        // Add more else if statements for other derived types as needed
        // e.g., else if (*type_info == typeid(another_derived_type)) { return as_ineq<another_derived_type>(); }
        else {
            throw std::runtime_error("Unknown supported type for inequality constraint: " + std::string(type_name));
        }
    }
    throw std::runtime_error("Unknown inequality constraint type: " + std::string(type_name));
}
} // namespace moto