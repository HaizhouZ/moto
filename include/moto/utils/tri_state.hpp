#ifndef MOTO_UTILS_TRI_STATE_HPP
#define MOTO_UTILS_TRI_STATE_HPP

namespace moto {
namespace utils {
enum tri_state {
    Unset = -1, ///< unset state
    False = 0,  ///< false state
    True = 1    ///< true state
};
struct tri_state_bool {
    tri_state value = Unset;
    tri_state_bool(bool v) : value(v ? True : False) {}
    tri_state_bool(tri_state v = Unset) : value(v) {}
    operator bool() const { return value == True; }
};
} // namespace utils
} // namespace moto

#endif // MOTO_UTILS_TRI_STATE_HPP