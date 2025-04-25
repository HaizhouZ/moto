#ifndef __SHOOTING_NODE__
#define __SHOOTING_NODE__

#include <array>
#include <mutex>
#include <stack>

#include <atri/ocp/approx.hpp>

namespace atri {

/**
 * @brief stacked approximation data
 * @note each std::vector contains the approximation data of functions in one
 * field
 */
class approx_sets_data {
    std::array<std::vector<approx_data_ptr_t>, field::num_func> ptr_;

  public:
    std::vector<approx_data_ptr_t> &operator[](size_t idx) {
        return ptr_[idx - field::num_sym];
    }
    const std::vector<approx_data_ptr_t> &operator[](size_t idx) const {
        return ptr_[idx - field::num_sym];
    }
    void swap(approx_sets_data &rhs) { ptr_.swap(rhs.ptr_); }
};

struct node_data {
    primal_data primal_data_;
    approx_sets_data approx_;
    node_data(approx_sets_data &&a, primal_data &&p)
        : approx_(std::move(a)), primal_data_(std::move(p)) {}
};

def_ptr(node_data);

/**
 * @brief data management. this class controls the data access and allocation.
 */
class data_mgr {
  private:
    struct data_pool : std::stack<node_data_ptr_t> {
        std::mutex mtx_;
    };

    def_ptr(data_pool);

    data_mgr() = default;

  public:
    // singleton
    static auto &get() {
        static data_mgr s_;
        return s_;
    }
    // make data for a expression set
    static node_data_ptr_t make_data(expr_sets_ptr_t expr_sets);

    void add_expr_sets(expr_sets_ptr_t exprs) {
        expr_sets_[exprs->uid_] = exprs;
        data_[exprs->uid_] = std::make_shared<data_pool>();
    }
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
    // the following are indexed by the uid of expr_sets
    // each is a problem formulation, with its own data
    std::unordered_map<size_t, expr_sets_ptr_t> expr_sets_;
    std::unordered_map<size_t, data_pool_ptr_t> data_;
};

/**
 * @brief shooting node in an OCP
 * @todo data collection/serialization/deserialization should be finished in
 * this node!
 */
class shooting_node {
  public:
    shooting_node(expr_sets_ptr_t formulation)
        : expr_sets_(formulation), mem_(data_mgr::get()) {
        data_ = mem_.acquire_data(expr_sets_);
    }

    ~shooting_node() { mem_.release_data(expr_sets_, data_); }

    void swap(shooting_node &p);
    void update_approximation();
    node_data_ptr_t data_;

  private:
    expr_sets_ptr_t expr_sets_;
    data_mgr &mem_;
};

def_ptr(shooting_node);
} // namespace atri

#endif /*__NODE_*/