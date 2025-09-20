#ifndef MOTO_OCP_CORE_DIRECTED_GRAPH_HPP
#define MOTO_OCP_CORE_DIRECTED_GRAPH_HPP

#include <list>
#include <moto/core/fwd.hpp>
#include <moto/core/parallel_job.hpp>
#include <moto/utils/movable_ptr.hpp>
#include <ranges>
#include <set>
#include <vector>

namespace moto {
namespace graph_types {
template <typename dtype, typename derived>
class node_base; // fwd

/**
 * @brief Base class for an edge in a directed graph
 *
 * @tparam node node type, must be derived from graph_types::node_base
 */
template <typename node>
struct edge_base {
    using node_type = node;
    movable_ptr<node> st; /// < start node of the edge
    movable_ptr<node> ed; /// < end node of the edge

    /// intermediate nodes, cloned from start node
    std::vector<node_type> nodes;
    /**
     * @brief construct a new edge from start to end node with a given length
     * it will clone the start node for each intermediate node
     * @param start start node
     * @param end end node
     * @param length number of appending nodes between start and end (excluding start and end)
     */
    edge_base(node *start, node *end, int length)
        : st(start), ed(end) {
        nodes.reserve(length); // reserve space for start and end nodes
        st->out_edges.emplace(this);
        ed->in_edges.emplace(this);
        if (length > 0) {
            while (length) {             // exclude ed node
                nodes.emplace_back(*st); // clone the start node
                --length;
            }
        }
    }
    ~edge_base() {
        if (st)
            st->out_edges.erase(this);
        if (ed)
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
template <typename dtype, typename derived>
struct node_base {
    using data_type = dtype;
    using edge_type = edge_base<derived>;
    std::set<edge_type *> in_edges;  ///< incoming edges
    std::set<edge_type *> out_edges; ///< outgoing edges
    size_t in_cnt{0}, out_cnt{0};    ///< counts of in and out edges
    movable_ptr<data_type> data_;    ///< pointer to the data
    /// necessary as proxy, e.g., node->(member of @ref node_base::data_type)
    data_type *operator->() { return data_; }
    operator data_type &() { return *data_; } ///< convert to data_type reference
    /// @brief default constructor, initializes @ref data_ to nullptr
    node_base() : data_(nullptr) {}
    /// @brief copy constructor (not really copied), initializes @ref data_ to nullptr
    node_base(const node_base &rhs) : data_(nullptr) {}
    /// @brief default move constructor
    node_base(node_base &&rhs) noexcept = default;
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
template <typename node>
class directed_graph {
  public:
    using data_type = typename node::data_type;
    using edge = graph_types::edge_base<node>;
    directed_graph(size_t n_jobs = MAX_THREADS)
        : n_jobs_(n_jobs) {
        unary_in_.reserve(100 * n_jobs);
        unary_view_.reserve(100 * n_jobs);
        binary_view_.reserve(100 * n_jobs);
    }
    size_t &n_jobs() { return n_jobs_; }                ///< get the number of jobs
    directed_graph(const directed_graph &rhs) = delete; ///< copy constructor is deleted
    /**
     * @brief add an edge from start node to end node with a given length
     * @ref graph_types::edge
     * @param st start node
     * @param ed end node
     * @param len length of the edge including start and end nodes
     * @return const auto& const reference to the added edge
     */
    void add_edge(node &st, node &ed, size_t len = 2, bool include_st = true, bool include_ed = true) {
        if (len < 2) {
            throw std::invalid_argument("Edge length must be no less than 2");
        }
        edges_.emplace_back(&st, &ed, len - (include_st + include_ed));
        // return edges_.back();
    }

    /**
     * @brief add a node to the graph
     *
     * @param d node to be added (movable)
     * @return node& reference to the added node
     */
    node &add(node &&d) {
        return nodes_.emplace_back(std::move(d));
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
            unary_in_.push_back(cur.data_);
            for (auto e : cur.out_edges) {
                for (auto &p : e->nodes) {
                    unary_in_.emplace_back(p.data_.get());
                }
            }
        }
    }

    /**
     * @brief Get the unordered flattened nodes list
     *
     * @return auto& the unordered flattened nodes list
     */
    auto &flatten_nodes() {
        unary_unordered_flatten();
        return unary_in_;
    }

    struct unary_view : std::vector<data_type *> {
        using base = std::vector<data_type *>;
        using base::base; ///< inherit constructors from std::vector
        bool update() {
            if (forward_mode) {
                forward_update();
            } else {
                backward_update();
            }
            return !this->empty();
        }

      private:
        friend class directed_graph; ///< allow directed_graph to access private members
        unary_view(node *head, node *tail, bool forward = true)
            : head_(head), tail_(tail), forward_mode(forward) {
            if (head_) {
                cur_nodes_.push_back(head_);
            } else if (tail_) {
                cur_nodes_.push_back(tail_);
            }
        }
        std::vector<node *> cur_nodes_;  ///< current edges in the graph
        std::vector<node *> next_nodes_; ///< next edges in the graph
        node *head_ = nullptr;           ///< head node of the graph
        node *tail_ = nullptr;           ///< tail node of the graph
        bool forward_mode = true;        ///< if true, apply unary function in forward direction
        void forward_update() {
            this->clear();
            while (!cur_nodes_.empty()) {
                for (auto n : cur_nodes_) {
                    // edge forward
                    this->emplace_back(n->data_);
                    for (auto out : n->out_edges) {
                        this->reserve(this->size() + out->nodes.size() + 1);
                        for (auto &out_node : out->nodes) {
                            this->emplace_back(out_node.data_);
                        }
                        if (--out->ed->in_cnt == 0) {
                            if (out->ed->out_edges.empty())
                                this->emplace_back(out->ed->data_);
                            else
                                next_nodes_.push_back(out->ed);
                        }
                    }
                }
                cur_nodes_.swap(next_nodes_);
                next_nodes_.clear();
            }
        }
        void backward_update() {
            this->clear();
            while (!cur_nodes_.empty()) {
                for (auto n : cur_nodes_) {
                    // edge backward
                    this->emplace_back(n->data_);
                    for (auto in : n->in_edges) {
                        this->reserve(this->size() + in->nodes.size() + 1);
                        for (auto it = in->nodes.rbegin(); it != in->nodes.rend(); ++it) {
                            this->emplace_back(it->data_);
                        }
                        if (--in->st->out_cnt == 0) {
                            if (in->st->in_cnt == 0)
                                this->emplace_back(in->st->data_);
                            else
                                next_nodes_.push_back(in->st);
                        }
                    }
                }
                cur_nodes_.swap(next_nodes_);
                next_nodes_.clear();
            }
        }
    };

    template <bool forward = true>
    unary_view &get_unary_view() {
        std::ranges::for_each(nodes_, [](auto &p) { p.reset_cnt(); });
        assert(head_ != nullptr && tail_ != nullptr && "head node must be set before applying unary function");
        return unary_view_ = unary_view(head_, tail_, forward);
    }

    unary_view forward_view() {
        return get_unary_view<true>();
    }

    unary_view backward_view() {
        return get_unary_view<false>();
    }

    struct binary_view : std::vector<std::pair<data_type *, data_type *>> {
        using base = std::vector<std::pair<data_type *, data_type *>>;
        using base::base; ///< inherit constructors from std::vector
        /// @brief update the binary view, i.e., update the current edges and next edges
        /// @return true if the binary view is updated, false if it is empty
        bool update() {
            if (forward_mode) {
                forward_update();
            } else {
                backward_update();
            }
            return !this->empty();
        }

      private:
        friend class directed_graph;     ///< allow directed_graph to access private members
        std::vector<edge *> cur_edges_;  ///< current edges in the graph
        std::vector<edge *> next_edges_; ///< next edges in the graph
        node *head_ = nullptr;           ///< head node of the graph
        node *tail_ = nullptr;           ///< tail node of the graph
        bool null_on_end_ = true;        ///< if true, append null to the end of the binary view
        bool forward_mode = true;
        binary_view(node *head, node *tail, bool forward = true, bool null_on_end = true)
            : head_(head), tail_(tail), forward_mode(forward), null_on_end_(null_on_end) {
            if (head_ && forward) {
                cur_edges_.assign(head_->out_edges.begin(), head_->out_edges.end());
            } else if (tail_ && !forward) {
                cur_edges_.assign(tail_->in_edges.begin(), tail_->in_edges.end());
            } else
                throw std::invalid_argument("head or tail node must be set before applying binary function");
        }
        void forward_update() {
            this->clear();
            while (!cur_edges_.empty()) {
                for (auto e : cur_edges_) {
                    // edge forward
                    data_type *cur = e->st->data_;
                    for (auto &next : e->nodes) {
                        this->emplace_back(cur, next.data_);
                        cur = next.data_;
                    }
                    this->emplace_back(cur, e->ed->data_);
                    if ((--e->ed->in_cnt) == 0 && !e->ed->out_edges.empty()) { // append ed to the list if no more in edges
                        next_edges_.insert(next_edges_.end(), e->ed->out_edges.begin(), e->ed->out_edges.end());
                    }
                }
                if (next_edges_.empty() && null_on_end_ && tail_) {
                    this->emplace_back(tail_->data_, nullptr); // append null to the end if head is set
                }
                cur_edges_.swap(next_edges_);
                next_edges_.clear();
            }
        }
        void backward_update() {
            this->clear();
            while (!cur_edges_.empty()) {
                for (size_t i = 0; i < cur_edges_.size(); ++i) {
                    for (auto e : cur_edges_) {
                        // edge forward
                        data_type *cur = e->ed->data_;
                        for (auto &prev : e->nodes | std::views::reverse) {
                            this->emplace_back(cur, prev.data_);
                            cur = prev.data_;
                        }
                        this->emplace_back(cur, e->st->data_);
                        if ((--e->st->out_cnt) == 0 && !e->st->in_edges.empty()) { // append st to the list if no more out edges
                            next_edges_.insert(next_edges_.end(), e->st->in_edges.begin(), e->st->in_edges.end());
                        }
                    }
                }
                if (next_edges_.empty() && null_on_end_ && head_) {
                    this->emplace_back(head_->data_, nullptr); // append null to the end if head is set
                }
                cur_edges_.swap(next_edges_);
                next_edges_.clear();
            }
        }
    };

    template <bool forward = true>
    binary_view &get_binary_view(bool null_on_end = false) {
        std::ranges::for_each(nodes_, [](auto &p) { p.reset_cnt(); });
        assert(head_ != nullptr && tail_ != nullptr && "head node must be set before applying binary function");
        return binary_view_ = binary_view(head_, tail_, forward, null_on_end);
    }

    binary_view &forward_binary_view(bool null_on_end = false) {
        return get_binary_view<true>(null_on_end);
    }

    binary_view &backward_binary_view(bool null_on_end = false) {
        return get_binary_view<false>(null_on_end);
    }

    /**
     * @brief apply unary function for all shooting nodes in parallel
     *
     * @param callback function [node]
     */
    template <typename callback_t>
    void for_each_parallel(callback_t &&callback) {
        constexpr bool is_unary = std::is_invocable_r_v<void, callback_t, data_type *>;
        constexpr bool is_unary_with_tid = std::is_invocable_r_v<void, callback_t, size_t, data_type *>;
        unary_unordered_flatten();
        if constexpr (is_unary_with_tid) {
            parallel_for(
                0, unary_in_.size(),
                [&callback, this](size_t tid, size_t i) { callback(tid, unary_in_[i]); },
                n_jobs_);
        } else if constexpr (is_unary) {
            parallel_for(
                0, unary_in_.size(),
                [&callback, this](size_t i) { callback(unary_in_[i]); },
                n_jobs_);
        } else {
            static_assert(false, "Callback function arity not supported in for_each_parallel()");
        }
    }

    template <bool forward = true, bool parallel = false, typename callback_t>
    void apply(callback_t &&callback, bool null_on_end = false) {
        constexpr bool is_unary = std::is_invocable_r_v<void, callback_t, data_type *>;
        constexpr bool is_unary_with_tid = std::is_invocable_r_v<void, callback_t, size_t, data_type *>;
        constexpr bool is_binary = std::is_invocable_r_v<void, callback_t, data_type *, data_type *>;
        constexpr bool is_binary_with_tid = std::is_invocable_r_v<void, callback_t, size_t, data_type *, data_type *>;
        if constexpr (is_unary || is_unary_with_tid) {
            if constexpr (forward) {
                unary_view_ = forward_view();
            } else {
                unary_view_ = backward_view();
            }
            if constexpr (is_unary_with_tid) {
                while (unary_view_.update()) {
                    sequential_for(
                        0, unary_view_.size(),
                        [&callback, this](size_t tid, size_t i) { callback(tid, unary_view_[i]); },
                        n_jobs_);
                }
            } else {
                while (unary_view_.update()) {
                    sequential_for(
                        0, unary_view_.size(),
                        [&callback, this](size_t i) { callback(unary_view_[i]); },
                        n_jobs_);
                }
            }
        } else if constexpr (is_binary || is_binary_with_tid) {
            if constexpr (forward) {
                binary_view_ = forward_binary_view(null_on_end);
            } else {
                binary_view_ = backward_binary_view(null_on_end);
            }
            while (binary_view_.update()) {
                if constexpr (is_binary_with_tid) {
                    auto job = [&callback, this](size_t tid, size_t i) {
                        callback(tid, binary_view_[i].first, binary_view_[i].second);
                    };
                    if constexpr (parallel)
                        parallel_for(0, binary_view_.size(), job, n_jobs_);
                    else
                        sequential_for(0, binary_view_.size(), job, n_jobs_);
                } else {
                    auto job = [&callback, this](size_t i) {
                        callback(binary_view_[i].first, binary_view_[i].second);
                    };
                    if constexpr (parallel)
                        parallel_for(0, binary_view_.size(), job, n_jobs_);
                    else
                        sequential_for(0, binary_view_.size(), job, n_jobs_);
                }
            }
        } else {
            static_assert(false, "Callback function arity not supported in apply()");
        }
    }

    template <bool parallel = false, typename callback_t>
    void apply_forward(callback_t &&callback, bool null_on_end = false) {
        apply<true, parallel>(std::forward<callback_t>(callback), null_on_end);
    } ///< apply function in forward direction

    template <typename callback_t>
    void apply_backward(callback_t &&callback, bool null_on_end = false) {
        apply<false>(std::forward<callback_t>(callback), null_on_end);
    } ///< apply function in backward direction

    // node &head() { return *head_; }
    // node &tail() { return *tail_; }
    // template <typename data_type>
    auto &nodes() {
        // std::vector<data_type *> nodes;
        // nodes.reserve(nodes_.size());
        // for (auto &n : nodes_) {
        //     nodes.push_back(static_cast<data_type *>(n->data_.get()));
        // }
        // return nodes;
        return nodes_;
    } ///< get the nodes in the graph

    auto &tail() { return *tail_; } ///< get the tail node of the graph

  private:
    size_t n_jobs_ = MAX_THREADS; ///< number of jobs to run in parallel
    node *head_ = nullptr;        /// < head node of the graph, i.e., the first node in the graph
    /// @todo: multiple tail
    node *tail_ = nullptr;  ///< tail node of the graph, i.e., the last node in the graph
    std::list<node> nodes_; ///< key nodes used to describe the graph
    std::list<edge> edges_; ///< edges used to connect the key nodes

    // temporary storage for unary and binary functions
    std::vector<data_type *> unary_in_; ///< data pointers for unary functions
    unary_view unary_view_;             ///< data pointers for unary functions
    binary_view binary_view_;           ///< data pointers for binary functions
    std::vector<edge *> cur_edges;      ///< edges for bfs current iteration
    std::vector<edge *> next_edges;     ///< edges for bfs next iteration
};
} // namespace moto

#endif // MOTO_OCP_CORE_DIRECTED_GRAPH_HPP