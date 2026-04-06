#ifndef MOTO_MODEL_GRAPH_MODEL_HPP
#define MOTO_MODEL_GRAPH_MODEL_HPP

#include <functional>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <moto/ocp/constr.hpp>
#include <moto/ocp/cost.hpp>
#include <moto/ocp/impl/func.hpp>
#include <moto/ocp/problem.hpp>
#include <moto/ocp/sym.hpp>

namespace moto {

class graph_model;

using model_node_ptr_t = node_ocp_ptr_t;
using model_edge_ptr_t = edge_ocp_ptr_t;

class graph_model {
  public:
    class storage_interface {
      public:
        virtual ~storage_interface() = default;

        // Reset any previously realized runtime graph state.
        virtual void clear() = 0;

        // Add one realized stage node and return a stable runtime handle id.
        // The handle id is later used by connect/set_head/set_tail.
        virtual size_t add_stage(const ocp_ptr_t &stage_ocp) = 0;

        // Connect two previously added stage handles.
        virtual void connect(size_t st_id, size_t ed_id) = 0;

        // Mark the unique runtime head node.
        virtual void set_head(size_t node_id) = 0;

        // Mark the unique runtime tail node.
        virtual void set_tail(size_t node_id) = 0;
    };
    using stage_builder_t = std::function<ocp_ptr_t(const ocp_ptr_t &)>;

    struct interval_compose_options {
        ocp::active_status_config source_config;
        bool materialize_sink_terms = false;
        bool include_terminal_sink_terms = false;
    };

    graph_model();

    void reserve(size_t node_capacity, size_t edge_capacity);

    model_node_ptr_t create_node(const node_ocp_ptr_t &base_prob = node_ocp::create());

    // Backward-compatible helper for bindings/examples: create a standalone edge
    // with auto-created source and sink nodes.
    model_edge_ptr_t create_edge(const edge_ocp_ptr_t &base_prob = edge_ocp::create());

    model_edge_ptr_t connect(const model_node_ptr_t &st,
                             const model_node_ptr_t &ed,
                             const edge_ocp_ptr_t &base_prob = edge_ocp::create());

    // Add a chain of n_edges edges from st to ed (creates n_edges-1 intermediate nodes).
    // Overloads: omit ed to auto-create the sink; omit both to auto-create source and sink.
    std::vector<model_edge_ptr_t> add_path(const model_node_ptr_t &st,
                                           const model_node_ptr_t &ed,
                                           size_t n_edges,
                                           const edge_ocp_ptr_t &base_prob = edge_ocp::create());

    std::vector<model_edge_ptr_t> add_path(const model_node_ptr_t &st,
                                           size_t n_edges,
                                           const edge_ocp_ptr_t &base_prob = edge_ocp::create());

    std::vector<model_edge_ptr_t> add_path(size_t n_edges,
                                           const edge_ocp_ptr_t &base_prob = edge_ocp::create());

    const model_node_ptr_t &node(size_t id) const;
    const model_edge_ptr_t &edge(size_t id) const;

    node_ocp_ptr_t compose_terminal(const model_node_ptr_t &node_h) const;

    // Python binding entry point — compose a single edge into a formulation.
    edge_ocp_ptr_t compose(const model_edge_ptr_t &edge_h) const;

    edge_ocp_ptr_t compose_interval(const model_edge_ptr_t &edge_h,
                                    const interval_compose_options &opts) const;

    std::vector<edge_ocp_ptr_t> compose_all() const;

    // Realize this modeled graph into a runtime graph adapter.
    //
    // Behavior:
    // 1) Calls graph.clear() exactly once at the beginning.
    // 2) Realizes exactly one runtime stage per modeled edge by composing that edge
    //    via compose_interval(...).
    // 3) For sink-final edges, sink terms (including terminal sink terms) are materialized.
    // 4) If stage_builder is provided, each composed stage is transformed by stage_builder
    //    before add_stage(...). If it returns null, realization throws.
    // 5) Runtime topology is constructed by connecting realized incoming-edge stages to
    //    realized outgoing-edge stages at each internal modeled node.
    // 6) Requires exactly one source path and one sink path among non-isolated nodes;
    //    otherwise throws.
    // 7) On success, sets head/tail and marks topology as up to date.
    //
    // Notes:
    // - Isolated modeled nodes (no incoming and no outgoing edge) are ignored.
    // - The handle returned by add_stage(...) is treated as an opaque id.
    void realize_into(storage_interface &graph, const stage_builder_t &stage_builder) const;

    // Convenience overload with identity stage mapping (no stage transformation).
    void realize_into(storage_interface &graph) const;

    bool topology_changed() const noexcept;
    size_t num_nodes() const noexcept;
    size_t num_edges() const noexcept;

  private:
    struct impl;
    static bool is_terminal_expr(const shared_expr &expr, field_t f);

    void validate_node(const model_node_ptr_t &node_h) const;
    void validate_edge(const model_edge_ptr_t &edge_h) const;

    std::vector<model_edge_ptr_t> add_path_impl(const model_node_ptr_t &st,
                                                const model_node_ptr_t &ed,
                                                size_t n_edges,
                                                const edge_ocp_ptr_t &base_prob);

    // Lower sink-node terms onto an already-composed edge formulation.
    // Non-terminal __cost terms are lowered unconditionally.
    // Terminal terms (any field) are lowered only when include_terminal is true.
    static void materialize_sink_terms(const node_ocp &sink_node,
                                       edge_ocp &composed,
                                       bool include_terminal);

    // Clone expr, substitute any __x args to __y (x_terminal → incoming y), cache in lowered_.
    static void lower_expr_into(const shared_expr &expr, edge_ocp &composed);

    std::shared_ptr<impl> state_;
};

} // namespace moto

#endif
