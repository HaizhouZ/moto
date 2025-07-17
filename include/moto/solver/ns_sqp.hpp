#ifndef __NS_SQP__
#define __NS_SQP__

#include <moto/ocp/impl/shooting_node.hpp>
#include <moto/solver/ipm/ipm_config.hpp>
#include <moto/solver/linesearch_config.hpp>
#include <moto/solver/ns_riccati/ns_riccati_data.hpp>

namespace moto {

struct ns_sqp {
    struct settings_t
        : public workspace_data_collection<solver::linesearch_config, ipm_impl::ipm_config> {
    } settings;

    struct node_type : public impl::shooting_node<ns_riccati::ns_node_data> {
        node_type(const ocp_ptr_t &prob)
            : impl::shooting_node<ns_riccati::ns_node_data>(prob) {}
        node_type(const node_type &rhs) = default;
        node_type(node_type &&rhs) = default;
    };
    void update(size_t n_iter);
    void forward();
    struct kkt_info {
        scalar_t objective = 0.;    // objective value
        scalar_t inf_prim_res = 0.; // primal residual (constraint violation)
        scalar_t inf_dual_res = 0.; // dual residual (stationary condition)
    };

    ns_sqp();

    directed_graph<node_type> graph_;
};

} // namespace moto

#endif