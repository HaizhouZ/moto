#ifndef MOTO_SOLVER_LINEAR_RUNTIME_GRAPH_HPP
#define MOTO_SOLVER_LINEAR_RUNTIME_GRAPH_HPP

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

    explicit linear_runtime_graph(size_t n_jobs = MAX_THREADS)
        : n_jobs_(normalize_parallel_jobs(n_jobs)) {}

    size_t n_jobs() const noexcept { return n_jobs_; }
    bool no_except() const noexcept { return no_except_; }
    void set_no_except(bool value) noexcept { no_except_ = value; }

    void clear() {
        nodes_.clear();
        ordered_.clear();
        order_dirty_ = true;
    }

    void reserve(size_t stage_capacity) {
        nodes_.reserve(stage_capacity);
        ordered_.reserve(stage_capacity);
    }

    node &add(node &&n) {
        nodes_.emplace_back(std::move(n));
        order_dirty_ = true;
        return nodes_.back();
    }

    auto &flatten_nodes() {
        ensure_order();
        return ordered_;
    }

  private:
    void ensure_order() {
        if (!order_dirty_) {
            return;
        }
        ordered_.clear();
        ordered_.reserve(nodes_.size());
        for (auto &stage : nodes_) {
            ordered_.push_back(&stage.payload());
        }
        order_dirty_ = false;
    }

    size_t n_jobs_ = MAX_THREADS;
    bool no_except_ = false;
    std::vector<node> nodes_;
    std::vector<data_type *> ordered_;
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
    auto at(size_t i) const noexcept { return std::tuple{(*a_)[i], (*b_)[i]}; }
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
    explicit adjacent_range(Graph &graph)
        : ordered_(&graph.flatten_nodes()), n_jobs_(graph.n_jobs()),
          no_except_(graph.no_except()) {}
    size_t size() const { return ordered_->size(); }
    auto at(size_t i) const noexcept {
        if constexpr (Forward) {
            return std::tuple{(*ordered_)[i],
                              i + 1 < ordered_->size() ? (*ordered_)[i + 1] : nullptr};
        } else {
            const size_t cur = ordered_->size() - 1 - i;
            return std::tuple{(*ordered_)[cur],
                              cur == 0 ? nullptr : (*ordered_)[cur - 1]};
        }
    }
    size_t n_jobs() const { return n_jobs_; }
    bool no_except() const { return no_except_; }
    std::vector<data_type *> *ordered_;
    size_t n_jobs_;
    bool no_except_;
};

template <graph_detail::graph_like Graph>
auto forward_edges(Graph &graph) {
    return adjacent_range<true, Graph>(graph);
}

template <graph_detail::graph_like Graph>
auto backward_edges(Graph &graph) {
    return adjacent_range<false, Graph>(graph);
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
