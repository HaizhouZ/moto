#ifndef MOTO_SOLVER_SOLVER_SETTING_HPP
#define MOTO_SOLVER_SOLVER_SETTING_HPP

#include <moto/core/fwd.hpp>

namespace moto {
namespace solver {

struct MOTO_ALIGN_NO_SHARING linesearch_config {
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
    void reset() {
        *this = linesearch_config();
    }
    void copy_from(const linesearch_config &other) {
        primal = other.primal;
        dual = other.dual;
        alpha_primal = other.alpha_primal;
        alpha_dual = other.alpha_dual;
    }
};

} // namespace solver
} // namespace moto

#endif // MOTO_SOLVER_SOLVER_SETTING_HPP