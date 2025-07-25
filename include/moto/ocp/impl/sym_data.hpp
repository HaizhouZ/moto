#ifndef MOTO_OCP_CORE_SYM_DATA_HPP
#define MOTO_OCP_CORE_SYM_DATA_HPP

#include <moto/ocp/sym.hpp>

namespace moto {
class ocp;
/**
 * @brief Symbolic data storage
 * stores the symbolic variables in a dense format
 * and provides access to the values of the symbolic variables
 *
 */
struct sym_data {
    /**
     * @brief Construct a new sym data object
     *
     * @param prob problem pointer, it will be used to get the dimensions of the symbolic variables
     */
    sym_data(ocp *prob);
    /// get the symbolic variable value of the sym
    vector_ref get(const sym &s);

    auto operator[](const sym &s) { return get(s); }
    auto operator[](sym *s) { return get(*s); }

    /// pointer to the problem, used to get dimensions of symbolic variables
    ocp *prob_;
    /// dense storage of symbolic variables, indexed by field
    std::array<vector, field::num_sym> value_;
    std::unordered_map<size_t, vector> usr_value_;
};

def_unique_ptr(sym_data);
} // namespace moto

#endif // MOTO_OCP_CORE_SYM_DATA_HPP