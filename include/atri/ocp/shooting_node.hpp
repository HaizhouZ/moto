#ifndef __SHOOTING_NODE__
#define __SHOOTING_NODE__

#include <atri/ocp/node_data.hpp>
#include <mutex>
#include <stack>

namespace atri {

/**
 * @brief data management. this class controls the data access and allocation.
 * 1. each data_mgr controls a node_data type
 * 2. data instances of same node_data type can have different exprs
 */
class data_mgr {
  private:
    struct data_pool : std::stack<node_data_ptr_t> {
        std::mutex mtx_;
    };
    using data_maker_func = std::function<node_data_ptr_t(expr_sets_ptr_t)>;

    def_ptr(data_pool);

    data_mgr() = default;
    data_mgr(data_mgr &) = delete;
    data_mgr(data_maker_func maker) : maker_(maker) {}

  public:
    // singleton
    template <typename data_type> static data_mgr &get() {
        static_assert(std::is_base_of<node_data, data_type>::value,
                      "data_type must be derived from node_data");
        static_assert(
            std::is_constructible<data_type, expr_sets_ptr_t>::value,
            "data_type must have a constructor that accepts expr_sets_ptr_t");

        data_maker_func maker = [](expr_sets_ptr_t exprs) {
            return std::static_pointer_cast<node_data>(
                std::make_shared<data_type>(exprs));
        };
        static data_mgr s_(maker);
        return s_;
    }
    // make data for a expression set
    static node_data_ptr_t make_data(expr_sets_ptr_t expr_sets);

    /**
     * @brief create a batch of data
     *
     * @param expr_sets the set of expression to be computed
     * @param N number of data instances. can be seen as stages
     */
    void create_data_batch(expr_sets_ptr_t expr_sets, size_t N);
    // thread-safe data access
    node_data_ptr_t acquire_data(expr_sets_ptr_t expr_sets);
    void release_data(expr_sets_ptr_t expr_sets, node_data_ptr_t data);

  private:
    data_maker_func maker_;
    // the following are indexed by the uid of expr_sets
    // each is a problem formulation, with its own data
    std::unordered_map<size_t, data_pool_ptr_t> data_;
};

/**
 * @brief shooting node in an OCP
 * @todo data collection/serialization/deserialization should be finished in
 * this node!
 */
struct shooting_node {
    shooting_node(expr_sets_ptr_t formulation, data_mgr &mem)
        : expr_sets_(formulation), mem_(mem) {
        data_ = mem_.acquire_data(expr_sets_);
    }

    ~shooting_node() { mem_.release_data(expr_sets_, data_); }

    void swap(shooting_node &p);
    void update_approximation();
    node_data_ptr_t data_;

    expr_sets_ptr_t expr_sets_;
    data_mgr &mem_;
};

def_ptr(shooting_node);
} // namespace atri

#endif /*__NODE_*/