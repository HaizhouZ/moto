#ifndef MOTO_CORE_WORKSPACE_DATA_HPP
#define MOTO_CORE_WORKSPACE_DATA_HPP

#include <concepts>
#include <moto/core/fwd.hpp>

namespace moto {
/**
 * @brief abstract wrapping class for workspace data
 *
 */
struct workspace_data {
    /**
     * @brief Get the data type from the workspace
     * @return lvalue ref to the data
     */
    template <typename T>
    T &as() {
        try {
            return dynamic_cast<T &>(*this);
        } catch (const std::bad_cast &e) {
            throw std::runtime_error(fmt::format("Invalid cast from workspace_data to {}", typeid(T).name()));
        }
    }

    virtual ~workspace_data() = default;
};

template <typename T>
concept has_worker_type = requires {
    typename T::worker_type; // Requires that T has a nested type alias named worker_type
};

template <typename... Ts>
class workspace_data_collection : public workspace_data, public Ts... {
  private:
    template <has_worker_type T>
    using get_worker_type = typename T::worker_type;

  public:

    struct worker : public workspace_data, public get_worker_type<Ts>... {
    };

    using worker_type = worker; ///< type of the worker

    workspace_data_collection() = default;

    workspace_data_collection(Ts &&...args) : Ts(std::forward<Ts>(args))... {
    }
};
} // namespace moto

#endif // MOTO_CORE_WORKSPACE_DATA_HPP