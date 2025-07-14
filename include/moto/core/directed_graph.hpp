#ifndef MOTO_OCP_CORE_DIRECTED_GRAPH_HPP
#define MOTO_OCP_CORE_DIRECTED_GRAPH_HPP

#include <atomic>
#include <moto/core/fwd.hpp>
#include <moto/core/parallel_job.hpp>
#include <ranges>
#include <set>
#include <vector>

namespace moto {
namespace graph_types {
template <typename dtype, typename derived, typename edge_type>
class node_base; // fwd

/**
 * @brief Base class for an edge in a directed graph
 *
 * @tparam node node type, must be derived from graph_types::node_base
 */
template <typename node>
struct edge_base {
    using node_type = node;
    def_unique_ptr_named(node, node_type);
    node *st; /// < start node of the edge
    node *ed; /// < end node of the edge

    /// intermediate nodes, cloned from start node
    std::vector<node_ptr_t> nodes;
    /**
     * @brief construct a new edge from start to end node with a given length
     * it will clone the start node for each intermediate node
     * @param start start node
     * @param end end node
     * @param length number of intermediate nodes
     */
    edge_base(node *start, node *end, int length)
        : st(start), ed(end) {
        st->out_edges.emplace(this);
        ed->in_edges.emplace(this);
        while (length--) {                                   // exclude ed node
            nodes.emplace_back(std::make_unique<node>(*st)); // clone the start node
        }
    }
    ~edge_base() {
        st->out_edges.erase(this);
        ed->in_edges.erase(this);
    }
};
/**
 * @brief Base class for a node in a directed graph
 *
 * @tparam dtype data type stored in the node (by pointer)
 * @tparam derived derived type of the node, used for CRTP
 * @tparam edge edge type, default is edge_base<derived>
 */
template <typename dtype, typename derived, typename edge = edge_base<derived>>
struct node_base {
    using data_type = dtype;
    using edge_type = edge;
    std::set<edge_type *> in_edges;            ///< incoming edges
    std::set<edge_type *> out_edges;           ///< outgoing edges
    std::atomic<size_t> in_cnt{0}, out_cnt{0}; ///< counts of in and out edges
    data_type *data_;                          ///< pointer to the data
    /// necessary as proxy, e.g., node->(member of @ref data_type)
    data_type *operator->() { return data_; }
    /// @brief default constructor, initializes data_ to nullptr
    node_base() : data_(nullptr) {}
    /// @brief copy constructor (not really copied), initializes data_ to nullptr
    node_base(const node_base &rhs) : data_(nullptr) {}
    /// @brief default move constructor
    node_base(node_base &&rhs) = default;
    /// @brief reset the counts of in and out edges
    void reset_cnt() {
        in_cnt = in_edges.size();
        out_cnt = out_edges.size();
    }
    virtual ~node_base() {}
};
} // namespace graph_types
/**
 * @brief A directed graph class that allows adding nodes and edges,
 * iterating over nodes, and applying unary or binary functions to the nodes.
 * @tparam node node type, must be derived from graph_types::node_base
 * @tparam edge edge type, must be derived from graph_types::edge_base, default is node::edge_type
 */
template <typename node,
          typename edge = node::edge_type>
class directed_graph {
  public:
    using data_type = typename node::data_type;
    /**
     * @brief add an edge from start node to end node with a given length
     * @ref graph_types::edge_base
     * @param st start node
     * @param ed end node
     * @param len length of the edge including start and end nodes
     * @return const auto& const reference to the added edge
     */
    const auto &add_edge(node &st, node &ed, size_t len = 2) {
        if (len < 2) {
            throw std::invalid_argument("Edge length must be no less than 2");
        }
        edges_.emplace_back(std::make_unique<edge>(&st, &ed, len - 2));
        return edges_.back();
    }

    /**
     * @brief add a node to the graph
     *
     * @param d node to be added (movable)
     * @return node& reference to the added node
     */
    node &add(node &&d) {
        nodes_.emplace_back(std::make_unique<node>(std::move(d)));
        return *nodes_.back();
    }

    node &set_tail(node &tail) { return *(tail_ = &tail); }
    node &set_head(node &head) { return *(head_ = &head); }
    /**
     * @brief get the flattened list of all nodes in the graph
     *
     */
    void unary_unordered_flatten() {
        unary_in_.clear();
        for (auto &cur : nodes_) {
            // fmt::print("Raw pointer of cur: {}\n", static_cast<void *>(cur.get()));
            unary_in_.push_back(cur->data_);
            for (auto e : cur->out_edges) {
                for (auto &p : e->nodes) {
                    unary_in_.emplace_back(p->data_);
                }
            }
        }
    }
    /**
     * @brief Get the unordered flattened nodes list
     * 
     * @return auto& the unordered flattened nodes list
     */
    auto& get_unordered_flattened_nodes() {
        unary_unordered_flatten();
        return unary_in_;
    }
    /**
     * @brief apply unary function for all shooting nodes in parallel
     *
     * @param callback function [node]
     */
    template <typename callback_t>
        requires std::invocable<callback_t, data_type *>
    void apply_all_unary_parallel(callback_t &&callback) {
        unary_unordered_flatten();
        parallel_for(0, unary_in_.size(), [this, &callback](size_t i) { callback(unary_in_[i]); });
    }
    /**
     * @brief apply unary function for all shooting nodes in parallel
     *
     * @param callback function [tid, node]
     */
    template <typename callback_t>
        requires std::invocable<callback_t, size_t, data_type *>
    void apply_all_unary_parallel(callback_t &&callback) {
        unary_unordered_flatten();
        parallel_for(0, unary_in_.size(), [this, &callback](size_t tid, size_t i) { callback(tid, unary_in_[i]); });
    }
    /**
     * @brief apply unary function for all shooting nodes sequentially
     *
     * @param callback function [node]
     */
    template <typename callback_t>
        requires std::invocable<callback_t, data_type *>
    void apply_all_unary_forward(callback_t &&callback) {
        unary_unordered_flatten();
        sequential_for(0, unary_in_.size(), [this, &callback](size_t i) { callback(unary_in_[i]); });
    }
    /**
     * @brief apply binary function for all shooting nodes sequentially or in parallel
     *
     * @param callback function [d, d+1]
     */
    template <bool parallel = false, bool tail_null_edge = false, typename callback_t>
        requires std::invocable<callback_t, data_type *, data_type *>
    void apply_all_binary_forward(callback_t &&callback) {
        std::ranges::for_each(nodes_, [](auto &p) { p->reset_cnt(); });
        // std::vector<std::pair<data_type *, data_type *>> binary_in_;
        // std::vector<edge *> cur_edges;                               // st nodes for this round
        // std::vector<edge *> next_edges;                              // st nodes for next round
        assert(head_ != nullptr && tail_ != nullptr && "head node must be set before applying binary function");
        cur_edges.assign(head_->out_edges.begin(), head_->out_edges.end()); // Ensure thread-local cur_edges is initialized
        next_edges.clear();                                                 // Clear thread-local next_edges
        binary_in_.clear();
        while (!cur_edges.empty()) {
            for (auto e : cur_edges) {
                // edge forward
                auto cur = e->st->data_;
                for (auto &next : e->nodes) {
                    binary_in_.emplace_back(cur, next->data_);
                    cur = next->data_;
                }
                binary_in_.emplace_back(cur, e->ed->data_);
                if ((--e->ed->in_cnt) == 0 && !e->ed->out_edges.empty()) { // append ed to the list if no more in edges
                    next_edges.insert(next_edges.end(), e->ed->out_edges.begin(), e->ed->out_edges.end());
                }
            }
            if constexpr (parallel) { // parallel by segments
                parallel_for(0, binary_in_.size(),
                             [&callback, this](size_t i) {
                                 callback(binary_in_[i].first, binary_in_[i].second);
                             });
            } else { // parallel by edges
                sequential_for(0, binary_in_.size(),
                               [&callback, this](size_t i) {
                                   callback(binary_in_[i].first, binary_in_[i].second);
                               });
            }
            cur_edges.swap(next_edges);
            next_edges.clear();
        }
        if constexpr (tail_null_edge) {
            callback(tail_->data_, nullptr);
        }
    }
    /**
     * @brief apply binary function for all shooting nodes backward sequentially
     *
     * @param callback function [d, d-1]
     */
    template <bool head_null_edge = true, typename callback_t>
        requires std::invocable<callback_t, data_type *, data_type *>
    void apply_all_binary_backward(callback_t &&callback) {
        std::for_each(nodes_.begin(), nodes_.end(), [](auto &p) { p->reset_cnt(); });
        // std::vector<std::pair<data_type *, data_type *>> binary_in_;
        // std::vector<edge *> cur_edges;                             // st nodes for this round
        // std::vector<edge *> next_edges;                            // st nodes for next round
        assert(head_ != nullptr && tail_ != nullptr && "head node must be set before applying binary function");
        cur_edges.assign(tail_->in_edges.begin(), tail_->in_edges.end()); // Ensure thread-local cur_edges is initialized
        next_edges.clear();                                               // Clear thread-local next_edges
        binary_in_.clear();
        while (!cur_edges.empty()) {
            // #pragma omp parallel for
            for (size_t i = 0; i < cur_edges.size(); ++i) {
                for (auto e : cur_edges) {
                    // edge forward
                    auto cur = e->ed->data_;
                    for (auto &prev : e->nodes | std::views::reverse) {
                        binary_in_.emplace_back(cur, prev->data_);
                        cur = prev->data_;
                    }
                    binary_in_.emplace_back(cur, e->st->data_);
                    if ((--e->st->out_cnt) == 0 && !e->st->in_edges.empty()) [[likely]] { // append st to the list if no more out edges
                        next_edges.insert(next_edges.end(), e->st->in_edges.begin(), e->st->in_edges.end());
                    }
                }
                sequential_for(0, binary_in_.size(),
                               [&callback, this](size_t i) {
                                   callback(binary_in_[i].first, binary_in_[i].second);
                               });
            }

            cur_edges.swap(next_edges);
            next_edges.clear();
        }
        if constexpr (head_null_edge) {
            callback(head_->data_, nullptr);
        }
    }

    node *head() { return head_; }
    node *tail() { return tail_; }

  private:
    def_unique_ptr(node);
    def_unique_ptr(edge);
    node *head_ = nullptr; /// < head node of the graph, i.e., the first node in the graph
    /// @todo: multiple tail
    node *tail_ = nullptr;          ///< tail node of the graph, i.e., the last node in the graph
    std::vector<node_ptr_t> nodes_; ///< key nodes used to describe the graph
    std::vector<edge_ptr_t> edges_; ///< edges used to connect the key nodes

    // temporary storage for unary and binary functions
    std::vector<data_type *> unary_in_;                          ///< data pointers for unary functions
    std::vector<std::pair<data_type *, data_type *>> binary_in_; ///< data pointers for binary functions
    std::vector<edge *> cur_edges;                               ///< edges for bfs current iteration
    std::vector<edge *> next_edges;                              ///< edges for bfs next iteration
};
} // namespace moto

#endif // MOTO_OCP_CORE_DIRECTED_GRAPH_HPP