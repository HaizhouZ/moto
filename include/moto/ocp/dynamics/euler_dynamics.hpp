#ifndef MOTO_OCP_EULER_DYNAMICS_HPP
#define MOTO_OCP_EULER_DYNAMICS_HPP

#include <moto/ocp/dynamics.hpp>

namespace moto {

template <typename T, size_t N>
using packed_retval = decltype(std::tuple_cat(std::array<T, N>()));

struct euler_impl; ///< forward declaration
struct euler : public func {

    generic_dynamics *operator->() const {
        return static_cast<generic_dynamics *>(func::operator->());
    }
    /// @brief add position variables
    packed_retval<var, 5> create_2nd_ord_vars(const std::string &name, size_t dim, bool semi_implicit = false);
    /// @brief create first order variables
    packed_retval<var, 3> create_1st_ord_vars(const std::string &name, size_t dim);

    void add_dt(scalar_t dt);   ///< set time step
    void add_dt(const var &dt); ///< set time variable

    euler(const std::string &name);

    using func::func;
};

} // namespace moto

#endif