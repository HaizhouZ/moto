#ifndef MOTO_SOLVER_SOLVER_SETTING_HPP
#define MOTO_SOLVER_SOLVER_SETTING_HPP

#include <moto/core/fwd.hpp>

namespace moto {
namespace solver {

struct MOTO_ALIGN_NO_SHARING linesearch_config {
    enum class dual_alpha_source : uint8_t {
        primal,
        dual,
    };

    // bound
    struct bounds {
        scalar_t alpha_max = 1.0; ///< max step size
        scalar_t alpha_min = 0.0; ///< min step size
        void merge_from(const bounds &other) {
            alpha_max = std::min(alpha_max, other.alpha_max);
            alpha_min = std::max(alpha_min, other.alpha_min);
            assert(alpha_max >= alpha_min);
        }
        void clip(scalar_t &alpha) const {
            if (alpha < alpha_min) {
                alpha = alpha_min;
            } else if (alpha > alpha_max) {
                alpha = alpha_max;
            }
        }
    } primal, dual;
    using worker_type = linesearch_config;
    scalar_t alpha_primal = 1.0; ///< primal step size
    scalar_t alpha_dual = 1.0;   ///< dual step size
    dual_alpha_source eq_dual_alpha_source = dual_alpha_source::primal;
    dual_alpha_source ineq_dual_alpha_source = dual_alpha_source::dual;
    bool update_alpha_dual = true;

    scalar_t dual_alpha_for_eq() const {
        return eq_dual_alpha_source == dual_alpha_source::primal ? alpha_primal : alpha_dual;
    }
    scalar_t dual_alpha_for_ineq() const {
        return ineq_dual_alpha_source == dual_alpha_source::primal ? alpha_primal : alpha_dual;
    }
    void reset() {
        alpha_primal = 1.0;
        alpha_dual = 1.0;
        primal = bounds();
        dual = bounds();
    }
    void copy_from(const linesearch_config &other) {
        primal = other.primal;
        dual = other.dual;
        alpha_primal = other.alpha_primal;
        alpha_dual = other.alpha_dual;
        eq_dual_alpha_source = other.eq_dual_alpha_source;
        ineq_dual_alpha_source = other.ineq_dual_alpha_source;
        update_alpha_dual = other.update_alpha_dual;
    }
};

} // namespace solver
} // namespace moto

#endif // MOTO_SOLVER_SOLVER_SETTING_HPP