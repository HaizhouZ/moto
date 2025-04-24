#ifndef __CLSQ__
#define __CLSQ__

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



} // namespace atri

#endif