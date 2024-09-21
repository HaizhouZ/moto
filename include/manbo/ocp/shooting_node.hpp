#ifndef __NODE__
#define __NODE__

#include <vector>
#include <memory>
#include <map>

#include <manbo/ocp/problem_formulation.hpp>
#include <manbo/core/approximation.hpp>

namespace manbo {
/**
 * @brief shooting node in an OCP
 * @todo data collection/serialization/deserialization should be finished in this node!
 */
class shootingNode {
   public:
    shootingNode(std::shared_ptr<problem_t> formulation)
        : problem_(formulation) {
    }
    void createData();
    void collect_data() {
        auto& constr_ = problem_->expr_[field_type::constr];
        for (size_t i = 0; i < constr_.size(); i++) {
            auto _c = std::static_pointer_cast<multivariate>(constr_[i]);
            if (_c->approx_level() == approx_type::first) {
                auto c = std::static_pointer_cast<first_approx>(_c);
                // c->jacobian();
            }
        }
    }

   private:
    std::vector<std::shared_ptr<shootingNode>> next_;
    std::shared_ptr<problem_t> problem_;
};
}  // namespace manbo

#endif /*__NODE_*/