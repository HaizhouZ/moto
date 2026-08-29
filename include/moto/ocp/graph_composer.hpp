#ifndef MOTO_OCP_GRAPH_COMPOSER_HPP
#define MOTO_OCP_GRAPH_COMPOSER_HPP

#include <array>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <moto/ocp/graph_model.hpp>

namespace moto {

class graph_composer {
  public:
    struct composed_interval {
        stage_ocp_ptr_t occurrence;
        ocp_ptr_t formulation;
    };

    struct interval_snapshot {
        size_t revision = 0;
        std::shared_ptr<const std::vector<composed_interval>> intervals;
    };

    interval_snapshot compose(const graph_model &model) const;

  private:
    enum class term_placement {
        direct,
        lower_x_to_y,
    };

    struct status_request {
        expr_handle expression;
        bool active = false;
    };

    using status_request_list = std::vector<status_request>;

    struct composition_key {
        std::array<stage_composition_identity_ptr_t, 4> identity;
        bool operator==(const composition_key &) const = default;
    };

    struct composition_key_hash {
        size_t operator()(const composition_key &key) const noexcept;
    };

    ocp_ptr_t compose_stage(const graph_model::topology_snapshot &topology,
                            size_t index) const;
    void append_placement(const stage_ocp_ptr_t &source, stage_expr_role role,
                          term_placement placement, const ocp_ptr_t &target,
                          status_request_list &status) const;
    void resolve_status(const status_request_list &status,
                        const ocp_ptr_t &target) const;

    mutable std::unordered_map<composition_key, ocp_ptr_t, composition_key_hash> entries_;
    mutable std::shared_ptr<const std::vector<composed_interval>> interval_cache_;
    mutable size_t cache_revision_ = 0;
    mutable std::mutex mutex_;
};

} // namespace moto

#endif
