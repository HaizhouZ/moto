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
 */
class data_mgr {
  private:
    struct data_pool : public std::stack<node_data_ptr_t> {
        std::mutex mtx_;
        data_pool() = default;
    };
    using make_data_func = std::function<node_data *(const problem_ptr_t &)>;
    node_data_ptr_t get_data(const problem_ptr_t &prob);

    data_mgr() = default;
    data_mgr(data_mgr &) = delete;
    data_mgr(make_data_func maker) : maker_(maker) {}

  public:
    // singleton
    template <typename data_type>
    static data_mgr &get() {
        static_assert(std::is_base_of<node_data, data_type>::value,
                      "data_type must be derived from node_data");
        static_assert(
            std::is_constructible<data_type, const problem_ptr_t &>::value,
            "data_type must have a constructor that accepts [const problem_ptr_t&]");

        make_data_func maker = [](const problem_ptr_t &prob) {
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
    void create_data_batch(const problem_ptr_t &problem, size_t N);
    // thread-safe data access
    node_data *acquire(const problem_ptr_t &problem);
    node_data *acquire(const node_data *rhs);
    void release(node_data *data);

  private:
    make_data_func maker_;
    // mapping [uid of problem, data pool]
    std::unordered_map<size_t, data_pool> data_;
};

} // namespace atri
#endif