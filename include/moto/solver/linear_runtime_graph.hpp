#ifndef MOTO_SOLVER_LINEAR_RUNTIME_GRAPH_HPP
#define MOTO_SOLVER_LINEAR_RUNTIME_GRAPH_HPP

#include <limits>
#include <stdexcept>
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

    template <typename callback_t>
    void for_each_parallel(callback_t &&callback) {
        ensure_order();
        constexpr bool with_tid = std::is_invocable_r_v<void, callback_t, size_t, data_type *>;
        constexpr bool unary = std::is_invocable_r_v<void, callback_t, data_type *>;
        if constexpr (with_tid) {
            parallel_for(
                0, ordered_.size(),
                [&](size_t tid, size_t i) { callback(tid, ordered_[i]); },
                n_jobs_, no_except_);
        } else if constexpr (unary) {
            parallel_for(
                0, ordered_.size(),
                [&](size_t i) { callback(ordered_[i]); },
                n_jobs_, no_except_);
        } else {
            static_assert(with_tid || unary, "unsupported callback arity in linear_runtime_graph::for_each_parallel");
        }
    }

    template <bool parallel = false, typename callback_t>
    void apply_forward(callback_t &&callback, bool null_on_end = false) {
        apply_binary<true, parallel>(std::forward<callback_t>(callback), null_on_end);
    }

    template <typename callback_t>
    void apply_backward(callback_t &&callback, bool null_on_end = false) {
        apply_binary<false, false>(std::forward<callback_t>(callback), null_on_end);
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

    template <bool forward, bool parallel, typename callback_t>
    void apply_binary(callback_t &&callback, bool null_on_end) {
        ensure_order();
        constexpr bool with_tid =
            std::is_invocable_r_v<void, callback_t, size_t, data_type *, data_type *>;
        constexpr bool binary =
            std::is_invocable_r_v<void, callback_t, data_type *, data_type *>;
        const size_t n_pairs = ordered_.empty()
                                   ? 0
                                   : (ordered_.size() - 1) + static_cast<size_t>(null_on_end);
        if constexpr (with_tid) {
            auto job = [&](size_t tid, size_t i) {
                if constexpr (forward) {
                    data_type *cur = ordered_[i];
                    data_type *next = (i + 1 < ordered_.size()) ? ordered_[i + 1] : nullptr;
                    callback(tid, cur, next);
                } else if (i + 1 == n_pairs && null_on_end) {
                    callback(tid, ordered_.front(), nullptr);
                } else {
                    const size_t cur_idx = ordered_.size() - 1 - i;
                    callback(tid, ordered_[cur_idx], ordered_[cur_idx - 1]);
                }
            };
            if constexpr (parallel) {
                parallel_for(0, n_pairs, job, n_jobs_, no_except_);
            } else {
                for (size_t i = 0; i < n_pairs; ++i) {
                    job(0, i);
                }
            }
        } else if constexpr (binary) {
            if constexpr (parallel) {
                parallel_for(
                    0, n_pairs,
                    [&](size_t i) {
                        if constexpr (forward) {
                            data_type *cur = ordered_[i];
                            data_type *next = (i + 1 < ordered_.size()) ? ordered_[i + 1] : nullptr;
                            callback(cur, next);
                        } else if (i + 1 == n_pairs && null_on_end) {
                            callback(ordered_.front(), nullptr);
                        } else {
                            const size_t cur_idx = ordered_.size() - 1 - i;
                            callback(ordered_[cur_idx], ordered_[cur_idx - 1]);
                        }
                    },
                    n_jobs_, no_except_);
            } else {
                for (size_t i = 0; i < n_pairs; ++i) {
                    if constexpr (forward) {
                        data_type *cur = ordered_[i];
                        data_type *next = (i + 1 < ordered_.size()) ? ordered_[i + 1] : nullptr;
                        callback(cur, next);
                    } else if (i + 1 == n_pairs && null_on_end) {
                        callback(ordered_.front(), nullptr);
                    } else {
                        const size_t cur_idx = ordered_.size() - 1 - i;
                        callback(ordered_[cur_idx], ordered_[cur_idx - 1]);
                    }
                }
            }
        } else {
            static_assert(with_tid || binary, "unsupported callback arity in linear_runtime_graph::apply_binary");
        }
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

} // namespace moto

#endif
