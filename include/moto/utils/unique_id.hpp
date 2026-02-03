#ifndef MOTO_UTILS_UNIQUE_ID_HPP
#define MOTO_UTILS_UNIQUE_ID_HPP

#include <atomic>

namespace moto {
namespace utils {
/// @brief thread-safe unique id generator for a given type
/// @tparam T
/// @note default uid value is invalid, set_inc() must be called to assign a new uid
template <typename T>
class unique_id {
  private:
    size_t uid_;

  public:
    constexpr static size_t uid_max = std::numeric_limits<size_t>::max();
    static std::atomic<size_t> max_uid; ///< maximum uid used for this type
    /// default constructor, assigns a new uid
    unique_id() : uid_(uid_max) {}
    void set_inc() { uid_ = max_uid++; } ///< set the uid to a new value
    /// copy constructor, assigns a new uid
    unique_id(const unique_id &rhs) : uid_(max_uid++) {}
    /// move constructor, assigns a new uid
    unique_id(unique_id &&rhs) noexcept : uid_(rhs.uid_) { rhs.uid_ = uid_max; }
    unique_id &operator=(const unique_id &rhs) {
        if (this != &rhs) { /// avoid self-assignment
            uid_ = max_uid++;
        }
        return *this;
    } ///< copy assignment operator, assigns a new uid
    unique_id &operator=(unique_id &&rhs) noexcept = default; ///< move assignment operator
    bool is_valid() const { return uid_ < uid_max; }          ///< check if the uid is valid
    operator size_t() const { return uid_; }                      ///< conversion to size_t operator
};

template <typename T>
inline auto format_as(const utils::unique_id<T> &uid) { return static_cast<size_t>(uid); } ///< format the unique id as a string

#define INIT_UID_(type) \
    template <>         \
    std::atomic<size_t> utils::unique_id<type>::max_uid = 0;

} // namespace utils
} // namespace moto
#endif