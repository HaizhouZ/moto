#ifndef MOTO_CORE_WORKSPACE_DATA_HPP
#define MOTO_CORE_WORKSPACE_DATA_HPP

#include <moto/core/fwd.hpp>
#include <concepts>

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
    T &get() {
        try {
            return dynamic_cast<T &>(*this);
        } catch (const std::bad_cast &e) {
            throw std::runtime_error(fmt::format("Invalid cast from workspace_data to {}", typeid(T).name()));
        }
    }
    /**
     * @brief Try to get the data type from the workspace
     * @return T* pointer to the data, nullptr if the cast fails
     */
    template <typename T>
    T *try_get() {
        T *ptr;
        try {
            ptr = dynamic_cast<T *>(this);
        } catch (const std::bad_cast &) {
            ptr = nullptr;
        } catch (...) {
            throw std::runtime_error(fmt::format("Unknown error in cast from workspace_data to {}", typeid(T).name()));
        }
        return ptr;
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
    using workspace_data::get;
    using workspace_data::try_get;

    struct worker : public workspace_data, public get_worker_type<Ts>... {
    };

    using worker_type = worker; ///< type of the worker

    workspace_data_collection() = default;

    workspace_data_collection(Ts &&...args) : Ts(std::forward<Ts>(args))... {
    }

    template <typename T>
        requires(std::disjunction_v<std::derived_from<T, Ts>...>)
    T &as() {
        return static_cast<T &>(*this);
    }
};
} // namespace moto

#endif // MOTO_CORE_WORKSPACE_DATA_HPP