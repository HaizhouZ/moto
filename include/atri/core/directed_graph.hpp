#ifndef ATRI_OCP_CORE_DIRECTED_GRAPH_HPP
#define ATRI_OCP_CORE_DIRECTED_GRAPH_HPP

#include <atomic>
#include <atri/core/fwd.hpp>
#include <atri/core/parallel_job.hpp>
#include <ranges>
#include <set>
#include <vector>

namespace atri {
template <typename value_type>
class directed_graph {
  public:
    using value_ptr_t = std::unique_ptr<value_type>;
    struct node;
    def_ptr(node);

    struct edge {
        node_ptr_t st;
        node_ptr_t ed;
        std::vector<value_type> nodes;
        size_t len; // length
        void resize(size_t new_len) { len = new_len; }
        edge(node_ptr_t start, node_ptr_t end, size_t length)
            : st(start), ed(end), len(length) {
            st->out_edges.emplace(this);
            ed->in_edges.emplace(this);
            while (--length) { // exclude ed node
                nodes.emplace_back(*st);
            }
        }
        ~edge() {
            st->out_edges.erase(this);
            ed->in_edges.erase(this);
        }
    };
    def_unique_ptr(edge);

    auto &add_edge(node_ptr_t st, node_ptr_t ed, size_t len) {
        edges_.push_back(std::make_unique<edge>(st, ed, len));
        return edges_.back();
    }
    // Sparse iterator
    struct node : public value_type {
        std::set<edge *> in_edges;
        std::set<edge *> out_edges;
        std::atomic<size_t> in_cnt, out_cnt;
        void reset_cnt() {
            in_cnt = in_edges.size();
            out_cnt = out_edges.size();
        }
        node(const value_type &n) : value_type(n) {}
        node(value_type &&n) : value_type(std::move(n)) {}
        node(const node &n) : node(static_cast<value_type>(n)) {}
    };

    // Add a node to the graph
    node_ptr_t add(const value_type &d) {
        nodes_.emplace_back(new node(d));
        return nodes_.back();
    }
    node_ptr_t add(value_type &&d) {
        nodes_.emplace_back(new node(std::move(d)));
        return nodes_.back();
    }

    void set_tail(node_ptr_t tail) { tail_ = tail; }
    void set_head(node_ptr_t head) { head_ = head; }
    auto begin() { return head_; }

    void unary_unordered_flatten() {
        unary_in_.clear();
        for (auto &cur : nodes_) {
            // fmt::print("Raw pointer of cur: {}\n", static_cast<void *>(cur.get()));
            unary_in_.push_back(cur.get());
            for (auto e : cur->out_edges) {
                for (auto &p : e->nodes) {
                    unary_in_.emplace_back(&p);
                }
            }
        }
    }

    template <typename callback_t>
        requires std::invocable<callback_t, value_type *>
    void apply_all_unary_parallel(callback_t &&callback) {
        unary_unordered_flatten();
        parallel_for(0, unary_in_.size(), [this, &callback](size_t i) { callback(unary_in_[i]); });
    }
    template <typename callback_t>
        requires std::invocable<callback_t, value_type *>
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
        requires std::invocable<callback_t, value_type *, value_type *>
    void apply_all_binary_forward(callback_t &&callback) {
        std::ranges::for_each(nodes_, [](node_ptr_t p) { p->reset_cnt(); });
        // std::vector<std::pair<value_type *, value_type *>> binary_in_;
        // std::vector<edge *> cur_edges;                               // st nodes for this round
        // std::vector<edge *> next_edges;                              // st nodes for next round
        cur_edges.assign(head_->out_edges.begin(), head_->out_edges.end()); // Ensure thread-local cur_edges is initialized
        next_edges.clear();                                                 // Clear thread-local next_edges
        binary_in_.clear();
        while (!cur_edges.empty()) {
            for (auto e : cur_edges) {
                // edge forward
                value_type *cur = e->st.get();
                for (auto &next : e->nodes) {
                    binary_in_.emplace_back(cur, &next);
                    cur = &next;
                }
                binary_in_.emplace_back(cur, e->ed.get());
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
            callback(tail_.get(), nullptr);
        }
    }
    /**
     * @brief apply binary function for all shooting nodes backward sequentially
     *
     * @param callback function [d, d-1]
     */
    template <bool head_null_edge = true, typename callback_t>
        requires std::invocable<callback_t, value_type *, value_type *>
    void apply_all_binary_backward(callback_t &&callback) {
        std::for_each(nodes_.begin(), nodes_.end(), [](node_ptr_t p) { p->reset_cnt(); });
        // std::vector<std::pair<value_type *, value_type *>> binary_in_;
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
                    value_type *cur = e->ed.get();
                    for (auto &prev : e->nodes | std::views::reverse) {
                        binary_in_.emplace_back(cur, &prev);
                        cur = &prev;
                    }
                    binary_in_.emplace_back(cur, e->st.get());
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
            callback(head_.get(), nullptr);
        }
    }

    node* head() {return head_.get();}
    node* tail() {return tail_.get();}

  private:
    node_ptr_t head_ = nullptr;
    /// @todo: multiple tail
    node_ptr_t tail_ = nullptr;
    std::vector<node_ptr_t> nodes_; // key nodes used to describe the graph
    std::vector<edge_ptr_t> edges_;

    std::vector<value_type *> unary_in_;
    std::vector<std::pair<value_type *, value_type *>> binary_in_;
    std::vector<edge *> cur_edges;  // st nodes for this round
    std::vector<edge *> next_edges; // st nodes for next round
};
} // namespace atri

#endif // ATRI_OCP_CORE_DIRECTED_GRAPH_HPP