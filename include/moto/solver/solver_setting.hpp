#ifndef MOTO_SOLVER_SOLVER_SETTING_HPP
#define MOTO_SOLVER_SOLVER_SETTING_HPP

#include <moto/core/fwd.hpp>
#include <new>

namespace moto {
namespace solver {

struct alignas(std::hardware_destructive_interference_size) line_search_cfg {
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
    scalar_t alpha_primal = 1.0; ///< primal step size
    scalar_t alpha_dual = 1.0;   ///< dual step size
    virtual ~line_search_cfg() = default; ///< virtual destructor to ensure polymorphism
};

struct solver_settings : public line_search_cfg {
    virtual ~solver_settings() = default; ///< virtual destructor to ensure polymorphism
};

} // namespace solver
} // namespace moto

#endif // MOTO_SOLVER_SOLVER_SETTING_HPP