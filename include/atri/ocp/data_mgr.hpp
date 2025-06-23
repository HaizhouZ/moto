#ifndef __ATRI_OCP_DATA_MGR__
#define __ATRI_OCP_DATA_MGR__

#include <atri/ocp/node_data.hpp>
#include <mutex>
#include <stack>

namespace atri {

/**
 * @brief data management. this class controls the data access and allocation.
 * 1. each data_mgr controls a data_type
 * 2. data instances of same data_type can have different prob
 * @details this class is dynamicsally allocated, i.e., it is a singleton for each data_type
 */
class data_mgr {
  private:
    struct data_pool : public std::stack<node_data_ptr_t> {
        std::mutex mtx_;
        data_pool() = default;
    };
    using make_data_func = std::function<node_data *(const ocp_ptr_t &)>;
    node_data_ptr_t get_data(const ocp_ptr_t &prob);

    data_mgr() = default;
    data_mgr(data_mgr &) = delete;
    data_mgr(make_data_func maker) : maker_(maker) {}

  public:
    /**
     * @brief get the data_mgr for a specific data_type
     *
     * @tparam data_type type of data to be managed, must be derived from node_data
     * @note data_type must have a constructor that accepts [const ocp_ptr_t&]
     * @return data_mgr& reference to the data_mgr instance of the data_type
     */
    template <typename data_type>
    static data_mgr &get() {
        static_assert(std::is_base_of<node_data, data_type>::value,
                      "data_type must be derived from node_data");
        static_assert(
            std::is_constructible<data_type, const ocp_ptr_t &>::value,
            "data_type must have a constructor that accepts [const ocp_ptr_t&]");

        make_data_func maker = [](const ocp_ptr_t &prob) {
            return new data_type(prob);
        };

        static data_mgr s_(maker);

        return s_;
    }

    /**
     * @brief create a batch of data
     *
     * @param problem the set of expression to be computed
     * @param N number of data instances. can be seen as stages
     */
    void create_data_batch(const ocp_ptr_t &problem, size_t N);
    /**
     * @brief acquire a data instance for a specific problem
     * @note this will create a new data instance if not found in the pool
     * @param problem the problem based on which the data is created
     * @return node_data* pointer to the data instance
     */
    node_data *acquire(const ocp_ptr_t &problem);
    /**
     * @brief acquire a data instance for a specific problem
     * @note this will create a new data instance if not found in the pool
     * @param rhs the node_data of which the problem is used to acquire the data
     * @return node_data* pointer to the data instance
     */
    node_data *acquire(const node_data *rhs);
    /**
     * @brief release a data instance back to the pool
     * @note this will not delete the data instance, but put it back to the pool
     * @param data pointer to the data instance to be released
     */
    void release(node_data *data);

  private:
    make_data_func maker_;
    /// mapping [uid of problem, data pool]
    std::unordered_map<size_t, data_pool> data_;
};

} // namespace atri
#endif