#ifndef MOTO_SOLVER_LINEAR_RUNTIME_GRAPH_HPP
#define MOTO_SOLVER_LINEAR_RUNTIME_GRAPH_HPP

#include <functional>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <moto/core/parallel_job.hpp>

namespace moto {

template <typename node>
class linear_runtime_graph {
  public:
    using data_type = typename node::data_type;

    struct edge_options {
        size_t len = 2;
        bool include_st = true;
        bool include_ed = true;
    };

    explicit linear_runtime_graph(size_t n_jobs = MAX_THREADS) : n_jobs_(n_jobs) {}

    size_t &n_jobs() { return n_jobs_; }
    bool &no_except() { return no_except_; }

    void clear() {
        nodes_.clear();
        next_.clear();
        prev_.clear();
        ordered_.clear();
        head_id_ = npos;
        tail_id_ = npos;
        order_dirty_ = true;
    }

    void reserve(size_t stage_capacity) {
        nodes_.reserve(stage_capacity);
        next_.reserve(stage_capacity);
        prev_.reserve(stage_capacity);
        ordered_.reserve(stage_capacity);
    }

    node &add(node &&n) {
        nodes_.emplace_back(std::move(n));
        nodes_.back().storage_id_ = nodes_.size() - 1;
        next_.push_back(npos);
        prev_.push_back(npos);
        order_dirty_ = true;
        return nodes_.back();
    }

    node &add_head(node &&n) {
        auto &out = add(std::move(n));
        return set_head(out);
    }

    node &add_tail(node &&n) {
        auto &out = add(std::move(n));
        return set_tail(out);
    }

    node &set_head(node &n) {
        head_id_ = node_id_of(n);
        order_dirty_ = true;
        return n;
    }

    node &set_tail(node &n) {
        tail_id_ = node_id_of(n);
        order_dirty_ = true;
        return n;
    }

    void add_edge(node &st, node &ed, size_t len = 2, bool include_st = true, bool include_ed = true) {
        connect(st, ed, {len, include_st, include_ed});
    }

    void connect(node &st, node &ed, edge_options opts = {}) {
        (void)opts;
        const size_t st_id = node_id_of(st);
        const size_t ed_id = node_id_of(ed);
        if (st_id == ed_id) {
            throw std::runtime_error("linear_runtime_graph does not support self-loops");
        }
        if (next_[st_id] != npos && next_[st_id] != ed_id) {
            throw std::runtime_error("linear_runtime_graph expects a single successor per node");
        }
        if (prev_[ed_id] != npos && prev_[ed_id] != st_id) {
            throw std::runtime_error("linear_runtime_graph expects a single predecessor per node");
        }
        next_[st_id] = ed_id;
        prev_[ed_id] = st_id;
        order_dirty_ = true;
    }

    node &insert_after(node &st, node &&next, edge_options opts = {}) {
        auto &out = add(std::move(next));
        connect(st, out, opts);
        return out;
    }

    std::vector<node *> add_path(std::vector<node> nodes,
                                 const std::vector<size_t> &steps,
                                 bool set_head = false,
                                 bool set_tail = false,
                                 bool include_st = true,
                                 bool include_ed = false) {
        if (nodes.empty()) {
            return {};
        }
        if (steps.size() + 1 != nodes.size()) {
            throw std::invalid_argument("linear_runtime_graph::add_path expects exactly one fewer edge-length than nodes");
        }
        std::vector<node *> added;
        added.reserve(nodes.size());
        for (auto &node_v : nodes) {
            auto &added_node = add(std::move(node_v));
            added.push_back(&added_node);
        }
        if (set_head) {
            this->set_head(*added.front());
        }
        if (set_tail) {
            this->set_tail(*added.back());
        }
        for (size_t i = 1; i < added.size(); ++i) {
            connect(*added[i - 1], *added[i], {steps[i - 1], include_st, include_ed});
        }
        return added;
    }

    auto &nodes() { return nodes_; }
    const auto &nodes() const { return nodes_; }

    auto &flatten_nodes() {
        ensure_order();
        return ordered_;
    }

  private:
    static constexpr size_t npos = std::numeric_limits<size_t>::max();

    size_t node_id_of(const node &n) const {
        if (n.storage_id_ == npos || n.storage_id_ >= nodes_.size()) {
            throw std::runtime_error("linear_runtime_graph cannot resolve node id");
        }
        return n.storage_id_;
    }

    void ensure_order() {
        if (!order_dirty_) {
            return;
        }
        ordered_.clear();
        if (nodes_.empty()) {
            head_id_ = tail_id_ = npos;
            order_dirty_ = false;
            return;
        }
        if (head_id_ == npos || tail_id_ == npos) {
            throw std::runtime_error("linear_runtime_graph requires both head and tail");
        }
        if (prev_[head_id_] != npos) {
            throw std::runtime_error("linear_runtime_graph head cannot have a predecessor");
        }
        if (next_[tail_id_] != npos) {
            throw std::runtime_error("linear_runtime_graph tail cannot have a successor");
        }

        ordered_.reserve(nodes_.size());
        std::vector<bool> seen(nodes_.size(), false);
        size_t cur = head_id_;
        while (cur != npos) {
            if (seen[cur]) {
                throw std::runtime_error("linear_runtime_graph detected a cycle");
            }
            seen[cur] = true;
            ordered_.push_back(nodes_[cur].data_.get());
            if (cur == tail_id_) {
                break;
            }
            cur = next_[cur];
        }
        if (ordered_.size() != nodes_.size() || !seen[tail_id_]) {
            throw std::runtime_error("linear_runtime_graph expects a single connected chain");
        }
        order_dirty_ = false;
    }

    size_t n_jobs_ = MAX_THREADS;
    bool no_except_ = false;
    std::vector<node> nodes_;
    std::vector<size_t> next_;
    std::vector<size_t> prev_;
    std::vector<data_type *> ordered_;
    size_t head_id_ = npos;
    size_t tail_id_ = npos;
    bool order_dirty_ = true;
};

namespace solver {

struct seq_t {};
struct par_t {};
inline constexpr seq_t seq{};
inline constexpr par_t par{};

namespace graph_detail {

template <typename T>
concept graph_like = requires(T &g) { typename T::data_type; g.flatten_nodes(); g.n_jobs(); g.no_except(); };

template <typename Callback, typename... Ptrs>
void invoke(Callback &&callback, size_t tid, Ptrs... ptrs) {
    if constexpr (std::is_invocable_r_v<void, Callback, size_t, Ptrs...>) {
        std::invoke(std::forward<Callback>(callback), tid, ptrs...);
    } else {
        static_assert(std::is_invocable_r_v<void, Callback, Ptrs...>,
                      "unsupported traversal callback arguments");
        std::invoke(std::forward<Callback>(callback), ptrs...);
    }
}

template <typename Callback, typename Item>
void invoke_item(Callback &&callback, size_t tid, Item &&item) {
    std::apply([&](auto... ptrs) { invoke(std::forward<Callback>(callback), tid, ptrs...); }, item);
}

} // namespace graph_detail

template <typename GraphA, typename GraphB>
struct zip_range {
    using data_type = typename GraphA::data_type;
    zip_range(GraphA &a, GraphB &b)
        : a_(&a.flatten_nodes()), b_(&b.flatten_nodes()),
          n_jobs_(a.n_jobs()), no_except_(a.no_except()) {
        if (a_->size() != b_->size()) {
            throw std::runtime_error("zip range size mismatch");
        }
    }
    size_t size() const { return a_->size(); }
    auto at(size_t i) const { return std::tuple{(*a_)[i], (*b_)[i]}; }
    size_t n_jobs() const { return n_jobs_; }
    bool no_except() const { return no_except_; }
    std::vector<typename GraphA::data_type *> *a_;
    std::vector<typename GraphB::data_type *> *b_;
    size_t n_jobs_;
    bool no_except_;
};

template <graph_detail::graph_like GraphA, graph_detail::graph_like GraphB>
auto zip(GraphA &a, GraphB &b) {
    return zip_range<GraphA, GraphB>(a, b);
}

template <bool Forward, typename Graph>
struct adjacent_range {
    using data_type = typename Graph::data_type;
    explicit adjacent_range(Graph &graph, bool null_on_end = false)
        : ordered_(&graph.flatten_nodes()), n_jobs_(graph.n_jobs()),
          no_except_(graph.no_except()), null_on_end_(null_on_end) {}
    size_t size() const {
        const size_t n = ordered_->size();
        return n == 0 ? 0 : (n - 1) + static_cast<size_t>(null_on_end_);
    }
    auto at(size_t i) const {
        if constexpr (Forward) {
            return std::tuple{(*ordered_)[i],
                              i + 1 < ordered_->size() ? (*ordered_)[i + 1] : nullptr};
        } else {
            if (i + 1 == size() && null_on_end_) {
                return std::tuple{ordered_->front(), static_cast<data_type *>(nullptr)};
            }
            const size_t cur = ordered_->size() - 1 - i;
            return std::tuple{(*ordered_)[cur], (*ordered_)[cur - 1]};
        }
    }
    size_t n_jobs() const { return n_jobs_; }
    bool no_except() const { return no_except_; }
    std::vector<data_type *> *ordered_;
    size_t n_jobs_;
    bool no_except_;
    bool null_on_end_;
};

template <graph_detail::graph_like Graph>
auto forward_edges(Graph &graph, bool null_on_end = false) {
    return adjacent_range<true, Graph>(graph, null_on_end);
}

template <graph_detail::graph_like Graph>
auto backward_edges(Graph &graph, bool null_on_end = false) {
    return adjacent_range<false, Graph>(graph, null_on_end);
}

template <typename Range, typename Callback>
    requires(!graph_detail::graph_like<std::remove_cvref_t<Range>>)
void for_each(seq_t, const Range &range, Callback &&callback) {
    for (size_t i = 0, n = range.size(); i < n; ++i) {
        graph_detail::invoke_item(callback, 0, range.at(i));
    }
}

template <typename Range, typename Callback>
    requires(!graph_detail::graph_like<std::remove_cvref_t<Range>>)
void for_each(par_t, const Range &range, Callback &&callback) {
    parallel_for(0, range.size(),
                 [&](size_t tid, size_t i) { graph_detail::invoke_item(callback, tid, range.at(i)); },
                 range.n_jobs(), range.no_except());
}

template <graph_detail::graph_like Graph, typename Callback>
void for_each(seq_t, Graph &graph, Callback &&callback) {
    auto &nodes = graph.flatten_nodes();
    for (auto *node : nodes) {
        graph_detail::invoke(callback, 0, node);
    }
}

template <graph_detail::graph_like Graph, typename Callback>
void for_each(par_t, Graph &graph, Callback &&callback) {
    auto &nodes = graph.flatten_nodes();
    parallel_for(0, nodes.size(),
                 [&](size_t tid, size_t i) { graph_detail::invoke(callback, tid, nodes[i]); },
                 graph.n_jobs(), graph.no_except());
}

} // namespace solver

} // namespace moto

#endif
