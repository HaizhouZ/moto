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
        std::vector<value_ptr_t> nodes;
        size_t len; // length
        void resize(size_t new_len) { len = new_len; }
        edge(node_ptr_t start, node_ptr_t end, size_t length)
            : st(start), ed(end), len(length) {
            st->out_edges.emplace(this);
            ed->in_edges.emplace(this);
            while (--length) { // exclude ed node
                nodes.push_back(std::make_unique<value_type>(*st));
            }
        }
        ~edge() {
            st->out_edges.erase(this);
            ed->in_edges.erase(this);
        }
    };
    def_unique_ptr(edge);

    void add_edge(node_ptr_t st, node_ptr_t ed, size_t len) {
        edges_.push_back(std::make_unique<edge>(st, ed, len));
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

    // return unordered flatten node list
    auto flatten() {
        std::vector<value_type *> out;
        for (auto cur : nodes_) {
            out.push_back(cur.get());
            for (auto e : cur->out_edges) {
                for (auto &p : e->nodes) {
                    out.emplace_back(p.get());
                }
            }
        }
        return out;
    }
    template <typename callback_t>
        requires std::invocable<callback_t, value_type *>
    void apply_all_unary_parallel(callback_t &&callback) {
        auto _nodes = flatten();
        parallel_for(0, _nodes.size(), [&_nodes, &callback](size_t i) { callback(_nodes[i]); });
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
        std::vector<edge *> cur_edges(head_->out_edges.begin(), head_->out_edges.end()); // st nodes for this round
        std::vector<edge *> next_edges;                                                  // st nodes for next round
        while (!cur_edges.empty()) {
            if constexpr (parallel) { // parallel by segments
                std::vector<std::pair<value_type *, value_type *>> in_;
                for (auto e : cur_edges) {
                    // edge forward
                    value_type *cur = e->ed.get();
                    for (auto &next : e->nodes | std::views::reverse) {
                        in_.emplace_back(cur, next.get());
                        cur = next.get();
                    }
                }
                parallel_for(0, in_.size(), [&callback, &in_](size_t i) { callback(in_[i].first, in_[i].second); });
                for (auto e : cur_edges) {
                    if ((--e->ed->in_cnt) == 0 && !e->ed->out_edges.empty()) { // append ed to the list if no more in edges
                        next_edges.insert(next_edges.end(), e->ed->out_edges.begin(), e->ed->out_edges.end());
                    }
                }
            } else { // parallel by edges
                parallel_for(0, cur_edges.size(), [&callback, &cur_edges, &next_edges](size_t i) {
                    auto e = cur_edges[i];
                    // edge forward
                    value_type *cur = e->st.get();
                    for (auto &next : e->nodes) {
                        callback(cur, next.get());
                        cur = next.get();
                    }
                    // last segment
                    callback(cur, e->ed.get());
                    if ((--e->ed->in_cnt) == 0 && !e->ed->out_edges.empty()) { // append ed to the list if no more in edges
                        next_edges.insert(next_edges.end(), e->ed->out_edges.begin(), e->ed->out_edges.end());
                    }
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
        std::vector<edge *> cur_edges(tail_->in_edges.begin(), tail_->in_edges.end()); // st nodes for this round
        std::vector<edge *> next_edges;                                                // st nodes for next round
        while (!cur_edges.empty()) {
            // #pragma omp parallel for
            for (size_t i = 0; i < cur_edges.size(); ++i) {
                auto e = cur_edges[i];
                // edge forward
                value_type *cur = e->ed.get();
                for (auto &next : e->nodes | std::views::reverse) {
                    callback(cur, next.get());
                    cur = next.get();
                }
                // last segment
                callback(cur, e->st.get());
                if ((--e->st->out_cnt) == 0 && !e->st->in_edges.empty()) [[likely]] { // append st to the list if no more out edges
                    next_edges.insert(next_edges.end(), e->st->in_edges.begin(), e->st->in_edges.end());
                }
            }

            cur_edges.swap(next_edges);
            next_edges.clear();
        }
        if constexpr (head_null_edge) {
            callback(head_.get(), nullptr);
        }
    }

  private:
    node_ptr_t head_ = nullptr;
    /// @todo: multiple tail
    node_ptr_t tail_ = nullptr;
    std::vector<node_ptr_t> nodes_; // key nodes used to describe the graph
    std::vector<edge_ptr_t> edges_;
};
} // namespace atri

#endif // ATRI_OCP_CORE_DIRECTED_GRAPH_HPP