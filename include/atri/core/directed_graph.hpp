#ifndef ATRI_OCP_CORE_DIRECTED_GRAPH_HPP
#define ATRI_OCP_CORE_DIRECTED_GRAPH_HPP

#include <atomic>
#include <atri/core/fwd.hpp>
#include <atri/core/parallel_job.hpp>
#include <ranges>
#include <set>
#include <vector>

namespace atri {
namespace directed_graph_types {
template <typename element_type, typename derived>
class node_type; // fwd

template <typename node_t>
struct edge_type {
    using node = node_t;
    def_unique_ptr(node);
    node *st;
    node *ed;
    std::vector<node_ptr_t> nodes;
    size_t len; // length
    void resize(size_t new_len) { len = new_len; }
    edge_type(node *start, node *end, size_t length)
        : st(start), ed(end), len(length) {
        st->out_edges.emplace(this);
        ed->in_edges.emplace(this);
        while (--length) {                                     // exclude ed node
            nodes.emplace_back(std::make_unique<node_t>(*st)); // clone the start node
        }
    }
    ~edge_type() {
        st->out_edges.erase(this);
        ed->in_edges.erase(this);
    }
};

template <typename T, typename derived>
struct node_type {
    using element_type = T;
    using edge = edge_type<derived>;
    std::set<edge *> in_edges;
    std::set<edge *> out_edges;
    std::atomic<size_t> in_cnt{0}, out_cnt{0};
    element_type *data_; // pointer to the data
    /// necessary as proxy, for user manipulation
    element_type *operator->() { return data_; }
    node_type() : data_(nullptr) {}
    node_type(const node_type &rhs) : data_(nullptr) {}
    node_type(node_type &&rhs) = default;
    void reset_cnt() {
        in_cnt = in_edges.size();
        out_cnt = out_edges.size();
    }
    virtual ~node_type() {}
};
} // namespace directed_graph_types
// Sparse iterator
template <typename node,
          typename edge = node::edge>
class directed_graph {
  public:
    using element_type = typename node::element_type;
    def_unique_ptr(node);
    def_unique_ptr(edge);
    const auto &add_edge(node &st, node &ed, size_t len) {
        edges_.emplace_back(std::make_unique<edge>(&st, &ed, len));
        return edges_.back();
    }

    // Add a node to the graph
    node &add(node &&d) {
        nodes_.emplace_back(std::make_unique<node>(std::move(d)));
        return *nodes_.back();
    }

    void set_tail(node &tail) { tail_ = &tail; }
    void set_head(node &head) { head_ = &head; }
    auto begin() { return head_; }

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

    template <typename callback_t>
        requires std::invocable<callback_t, element_type *>
    void apply_all_unary_parallel(callback_t &&callback) {
        unary_unordered_flatten();
        parallel_for(0, unary_in_.size(), [this, &callback](size_t i) { callback(unary_in_[i]); });
    }
    template <typename callback_t>
        requires std::invocable<callback_t, element_type *>
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
        requires std::invocable<callback_t, element_type *, element_type *>
    void apply_all_binary_forward(callback_t &&callback) {
        std::ranges::for_each(nodes_, [](auto &p) { p->reset_cnt(); });
        // std::vector<std::pair<element_type *, element_type *>> binary_in_;
        // std::vector<edge *> cur_edges;                               // st nodes for this round
        // std::vector<edge *> next_edges;                              // st nodes for next round
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
        requires std::invocable<callback_t, element_type *, element_type *>
    void apply_all_binary_backward(callback_t &&callback) {
        std::for_each(nodes_.begin(), nodes_.end(), [](auto &p) { p->reset_cnt(); });
        // std::vector<std::pair<element_type *, element_type *>> binary_in_;
        // std::vector<edge *> cur_edges;                             // st nodes for this round
        // std::vector<edge *> next_edges;                            // st nodes for next round
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
    node *head_ = nullptr;
    /// @todo: multiple tail
    node *tail_ = nullptr;
    std::vector<node_ptr_t> nodes_; // key nodes used to describe the graph
    std::vector<edge_ptr_t> edges_;

    std::vector<element_type *> unary_in_;
    std::vector<std::pair<element_type *, element_type *>> binary_in_;
    std::vector<edge *> cur_edges;  // st nodes for this round
    std::vector<edge *> next_edges; // st nodes for next round
};
} // namespace atri

#endif // ATRI_OCP_CORE_DIRECTED_GRAPH_HPP