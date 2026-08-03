#ifndef MOTO_OCP_GRAPH_COMPOSER_HPP
#define MOTO_OCP_GRAPH_COMPOSER_HPP

#include <memory>
#include <mutex>
#include <vector>

#include <moto/ocp/graph_model.hpp>

namespace moto {

class graph_composer {
  public:
    struct interval_snapshot {
        size_t revision = 0;
        std::shared_ptr<const std::vector<ocp_ptr_t>> intervals;
    };

    interval_snapshot compose(const graph_model &model) const;

  private:
    enum class term_placement {
        direct,
        lower_x_to_y,
    };

    struct dependency {
        const stage_ocp *stage = nullptr;
        size_t revision = 0;
        stage_expr_role role = stage_expr_role::interval;
        bool operator==(const dependency &) const = default;
    };

    struct cache_entry {
        std::vector<dependency> dependencies;
        ocp_ptr_t composed;
    };

    std::vector<dependency> dependencies(const graph_model::interval_record &record) const;
    ocp_ptr_t compose_stage(const graph_model::interval_record &record) const;
    void append_role_terms(const stage_ocp_ptr_t &source, stage_expr_role role, const ocp_ptr_t &target,
                           term_placement placement) const;
    void append_node_terms(const node_view &node, const ocp_ptr_t &target, term_placement placement) const;

    mutable std::vector<cache_entry> entries_;
    mutable std::shared_ptr<const std::vector<ocp_ptr_t>> interval_cache_;
    mutable size_t cache_revision_ = 0;
    mutable std::mutex mutex_;
};

} // namespace moto

#endif
