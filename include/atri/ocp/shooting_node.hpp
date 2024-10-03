#ifndef __SHOOTING_NODE__
#define __SHOOTING_NODE__

#include <mutex>
#include <stack>
#include <array>

#include <atri/ocp/approximation.hpp>

namespace atri {

typedef std::array<std::vector<approx_data_ptr_t>, field::num_func> stacked_approx_ptr;

class mem_mgr {
   private:
    template <typename data_type>
    struct lck_data {
        std::mutex mtx_;
        std::stack<data_type> data_;
    };

    mem_mgr() = default;

   public:
    static auto& get() {
        static mem_mgr s_;
        return s_;
    }

    void add_expr_collection(expr_collection_ptr_t expr_collection) {
        expr_collections_[expr_collection->uid_] = expr_collection;
        approx_[expr_collection->uid_] = std::make_shared<lck_data<stacked_approx_ptr>>();
    }

    static stacked_approx_ptr make_approx_data(expr_collection_ptr_t expr_collection);

    void create_data_batch(expr_collection_ptr_t expr_collection, size_t N);
    stacked_approx_ptr acquire_approx(expr_collection_ptr_t expr_collection);
    void release_approx(expr_collection_ptr_t expr_collection, stacked_approx_ptr&& data);

   private:
    std::unordered_map<size_t, expr_collection_ptr_t> expr_collections_;
    std::unordered_map<size_t, std::shared_ptr<lck_data<stacked_approx_ptr>>> approx_;
};

/**
 * @brief shooting node in an OCP
 * @todo data collection/serialization/deserialization should be finished in this node!
 */
class shooting_node {
   public:
    shooting_node(expr_collection_ptr_t formulation)
        : expr_collection_(formulation), mem_(mem_mgr::get()) {
        approx_ = std::move(mem_.acquire_approx(expr_collection_));
    }

    ~shooting_node() {
        mem_.release_approx(expr_collection_, std::move(approx_));
    }

    void swap(shooting_node& p);
    void collect_data();

   private:
    expr_collection_ptr_t expr_collection_;
    mem_mgr& mem_;
    stacked_approx_ptr approx_;
    primal_data_ptr_t primal_data_;
};
}  // namespace atri

#endif /*__NODE_*/