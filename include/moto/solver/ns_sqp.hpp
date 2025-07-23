#ifndef __NS_SQP__
#define __NS_SQP__

#include <moto/ocp/impl/shooting_node.hpp>
#include <moto/solver/ipm/ipm_config.hpp>
#include <moto/solver/linesearch_config.hpp>
#include <moto/solver/ns_riccati/ns_riccati_data.hpp>

namespace moto {

struct ns_sqp {
    struct settings_t
        : public workspace_data_collection<solver::linesearch_config, solver::ipm_config> {
    } settings;
    // using node_base = ;
    using ns_node_data = solver::ns_riccati::ns_node_data;
    struct data : public composed_data<node_data, ns_node_data> {
        using base = composed_data<node_data, ns_node_data>;
        data(const ocp_ptr_t &prob)
            : base(node_data(prob), ns_node_data(node_data::sym_.get(), node_data::dense_.get())) {}
        data(data &&rhs) = default;
        static void update_approx(data *d) {
            d->update_approximation();
        }
    };
    void update(size_t n_iter);
    void forward();
    struct kkt_info {
        scalar_t objective = 0.;    // objective value
        scalar_t inf_prim_res = 0.; // primal residual (constraint violation)
        scalar_t inf_dual_res = 0.; // dual residual (stationary condition)
        scalar_t inf_comp_res = 0.; // (inequality) complementarity residual
    };

    ns_sqp();
    using node_type = impl::shooting_node<data>;
    directed_graph<node_type> graph_;
};

} // namespace moto

#endif