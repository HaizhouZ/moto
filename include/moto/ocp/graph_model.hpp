#ifndef MOTO_MODEL_GRAPH_MODEL_HPP
#define MOTO_MODEL_GRAPH_MODEL_HPP

#include <atomic>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

#include <moto/ocp/problem.hpp>

namespace moto {

struct ns_sqp;

class graph_model {
  public:
    graph_model();

    node_view start_node() const;
    std::vector<stage_ocp_ptr_t> add_stage(const stage_ocp_ptr_t &stage,
                                           size_t n_stages);
    std::vector<stage_ocp_ptr_t> add_stages(const node_view &start_node,
                                            const stage_ocp_ptr_t &stage,
                                            size_t n_stages);

  private:
    friend struct ns_sqp;

    struct revision_state {
        std::atomic<size_t> revision = 1;
    };

    struct interval_record {
        stage_ocp_ptr_t stage;
        node_view start_boundary_view;
        std::vector<node_view> end_boundary_views;
    };

    struct interval_snapshot {
        size_t revision;
        std::shared_ptr<const std::vector<ocp_ptr_t>> intervals;
    };

    struct stage_chain {
        std::vector<interval_record> records;
        std::vector<stage_ocp_ptr_t> stages;
        node_view tail;
    };

    interval_snapshot composed_intervals() const;
    size_t revision() const noexcept { return revision_state_->revision.load(std::memory_order_acquire); }
    ocp_ptr_t compose_stage(const interval_record &record) const;
    enum class term_placement {
        direct,
        lower_x_to_y,
    };
    void append_role_terms(const stage_ocp_ptr_t &source,
                           stage_expr_role role,
                           const ocp_ptr_t &target,
                           term_placement placement) const;
    void append_node_terms(const node_view &node,
                           const ocp_ptr_t &target,
                           term_placement placement) const;
    stage_chain build_stage_chain(const node_view &start_node,
                                  const stage_ocp_ptr_t &stage,
                                  size_t n_stages);
    std::vector<stage_ocp_ptr_t> commit_stage_chain(stage_chain &&chain,
                                                    std::optional<size_t> incoming_boundary_index,
                                                    bool advances_tail);
    void validate_stage_chain_input(const node_view &start_node,
                                    const stage_ocp_ptr_t &stage,
                                    size_t n_stages) const;
    void add_end_boundary_view(interval_record &record, const node_view &node) const;
    std::optional<size_t> find_incoming_boundary_index(const node_view &node,
                                                       const char *missing_message) const;
    void invalidate();
    void attach_graph_callback(const stage_ocp_ptr_t &stage);
    bool same_node(const node_view &lhs, const node_view &rhs) const;

    std::shared_ptr<revision_state> revision_state_ = std::make_shared<revision_state>();
    stage_ocp_ptr_t start_stage_ = stage_ocp::create();
    node_view tail_node_ = start_stage_->st();
    std::vector<interval_record> intervals_;
    mutable std::shared_ptr<const std::vector<ocp_ptr_t>> interval_cache_;
    mutable size_t interval_cache_revision_ = 0;
    mutable std::mutex graph_state_mutex_;
};

} // namespace moto

#endif
