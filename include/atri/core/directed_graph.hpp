#ifndef ATRI_OCP_CORE_DIRECTED_GRAPH_HPP
#define ATRI_OCP_CORE_DIRECTED_GRAPH_HPP

#include <atomic>
#include <atri/core/fwd.hpp>
#include <set>
#include <vector>

namespace atri {
template <typename value_type>
class directed_graph {
  public:
    using value_ptr_t = std::shared_ptr<value_type>;
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
                nodes.emplace_back(std::make_shared<value_type>(*st));
            }
        }
        ~edge() {
            st->out_edges.erase(this);
            ed->in_edges.erase(this);
        }
    };
    def_ptr(edge);

    void add_edge(node_ptr_t st, node_ptr_t ed, size_t len) {
        edges_.emplace_back(new edge(st, ed, len));
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
        node(const node &n) : node(static_cast<value_type>(n)) {}
    };

    // Add a node to the graph
    node_ptr_t add(const value_type &d) {
        nodes_.emplace_back(new node(d));
        return nodes_.back();
    }

    void set_tail(node_ptr_t tail) { tail_ = tail; }
    void set_head(node_ptr_t head) { head_ = head; }
    auto begin() { return head_; }

    // return unordered flatten node list
    auto flatten() {
        std::vector<value_ptr_t> out;
        for (auto cur : nodes_) {
            out.push_back(cur);
            for (auto e : cur->out_edges) {
                out.insert(out.end(), e->nodes.begin(), e->nodes.end());
            }
        }
        return out;
    }

    void apply_all_unary(std::function<void(value_ptr_t)> callback) {
#pragma omp parallel
        for (auto cur : flatten())
            callback(cur);
    }
    /**
     * @brief apply binary function for all shooting nodes sequentially
     *
     * @param callback function [d, d+1]
     */
    void apply_all_binary_forward(std::function<void(value_ptr_t, value_ptr_t)> callback) {
        std::for_each(nodes_.begin(), nodes_.end(), [](node_ptr_t p) { p->reset_cnt(); });
        std::vector<edge *> cur_edges(head_->out_edges.begin(), head_->out_edges.end()); // st nodes for this round
        std::vector<edge *> next_edges;                                                  // st nodes for next round
        while (!cur_edges.empty()) {
#pragma omp parallel
            for (auto e : cur_edges) {
                // edge forward
                value_ptr_t cur = e->st;
                for (auto next : e->nodes) {
                    callback(cur, next);
                    cur = next;
                }
                // last segment
                callback(cur, e->ed);
                if ((--e->ed->in_cnt) == 0 && !e->ed->out_edges.empty()) { // append ed to the list if no more in edges
                    next_edges.insert(next_edges.end(), e->ed->out_edges.begin(), e->ed->out_edges.end());
                }
            }

            cur_edges.swap(next_edges);
            next_edges.clear();
        }
    }
    /**
     * @brief apply binary function for all shooting nodes backward sequentially
     *
     * @param callback function [d, d-1]
     */
    void apply_all_binary_backward(std::function<void(value_ptr_t, value_ptr_t)> callback) {
        std::for_each(nodes_.begin(), nodes_.end(), [](node_ptr_t p) { p->reset_cnt(); });
        std::vector<edge *> cur_edges(tail_->in_edges.begin(), tail_->in_edges.end()); // st nodes for this round
        std::vector<edge *> next_edges;                                                // st nodes for next round
        while (!cur_edges.empty()) {
#pragma omp parallel
            for (auto e : cur_edges) {
                // edge forward
                value_ptr_t cur = e->ed;
                for (auto next = e->nodes.rbegin(); next != e->nodes.rend(); ++next) {
                    callback(cur, *next);
                    cur = *next;
                }
                // last segment
                callback(cur, e->st);
                if ((--e->st->out_cnt) == 0 && !e->st->in_edges.empty()) { // append st to the list if no more out edges
                    next_edges.insert(next_edges.end(), e->st->in_edges.begin(), e->st->in_edges.end());
                }
            }

            cur_edges.swap(next_edges);
            next_edges.clear();
        }
    }

  private:
    node_ptr_t head_;
    node_ptr_t tail_;               // todo: multiple tail
    std::vector<node_ptr_t> nodes_; // key nodes used to describe the graph
    std::vector<edge_ptr_t> edges_;
};
} // namespace atri

#endif // ATRI_OCP_CORE_DIRECTED_GRAPH_HPP