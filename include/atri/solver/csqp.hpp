#ifndef __CSQP__
#define __CSQP__

#include <atri/ocp/shooting_node.hpp>
#include <list>

namespace atri {
/// @todo replace with a graph
using ocp_graph = std::vector<shooting_node_ptr_t>;

class solver_base {
  protected:
    std::shared_ptr<ocp_graph> graph_;

  public:
    virtual ~solver_base() = default;

    solver_base(std::shared_ptr<ocp_graph> graph) : graph_(graph) {}

    virtual void backward_pass() {}
    // update primal variables
    virtual void forward_pass() {}

    // update the derivatives of the primal variables
    void update_derivatives() {
#pragma omp parallel for
        for (int i = 0; i < graph_->size(); i++) {
            auto s = graph_->at(i);
            s->update_approximation();
        }
    }
};

struct value_func_data {};

struct nullspace_riccati_data : public node_data {
    value_func_data value_func_;
    nullspace_riccati_data(expr_sets_ptr_t exprs) : node_data(exprs) {}
};

def_ptr(nullspace_riccati_data);

class nullspace_riccati_solver {
    data_mgr &mem_;

  public:
    nullspace_riccati_solver()
        : mem_(data_mgr::get<nullspace_riccati_data>()) {}
};

} // namespace atri

#endif