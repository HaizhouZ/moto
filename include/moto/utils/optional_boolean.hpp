#ifndef MOTO_UTILS_OPTIONAL_BOOL_HPP
#define MOTO_UTILS_OPTIONAL_BOOL_HPP

namespace moto {
namespace utils {
/**
 * @brief Optional boolean type to represent a boolean value that can be unset.
 * This is useful for cases where a boolean value may not be applicable or known.
 * @note default value is Unset, which indicates that the boolean value has not been set.
 */
struct optional_bool {
    enum val_type {
        Unset = -1, ///< unset state
        False = 0,  ///< false state
        True = 1    ///< true state
    };
    val_type value = Unset;
    optional_bool(bool v) : value(v ? True : False) {}
    optional_bool(val_type v = Unset) : value(v) {}
    operator bool() const { return value == True; }
    bool is_set() const { return value != Unset; } ///< check if the value is set
    bool operator==(const optional_bool &rhs) const {
        return value == rhs.value;
    }
    bool operator==(val_type rhs) const {
        return value == rhs;
    }
    bool operator!=(const optional_bool &rhs) const {
        return value != rhs.value;
    }
};
} // namespace utils
} // namespace moto

#endif // MOTO_UTILS_OPTIONAL_BOOL_HPP