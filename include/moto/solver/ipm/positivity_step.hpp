#pragma once

#include <moto/solver/linesearch_config.hpp>

namespace moto::solver::positivity {

inline scalar_t alpha_max(const vector &value,
                          const vector &step,
                          scalar_t fraction_to_boundary = scalar_t(0.995)) {
    scalar_t alpha = 1.;
    for (Eigen::Index i = 0; i < value.size(); ++i) {
        if (step(i) < 0.) {
            alpha = std::min(alpha, -fraction_to_boundary * value(i) / step(i));
        }
    }
    return std::max(alpha, scalar_t(0.));
}

template <typename DualValue>
inline void backup_pair(const vector &value,
                        vector &value_backup,
                        const DualValue &dual,
                        vector &dual_backup) {
    value_backup = value;
    dual_backup = dual;
}

template <typename DualValue>
inline void restore_pair(vector &value,
                         const vector &value_backup,
                         DualValue &&dual,
                         const vector &dual_backup) {
    value = value_backup;
    dual = dual_backup;
}

template <typename DualValue, typename DualStep>
inline void apply_pair_step(vector &value,
                            const vector &value_step,
                            scalar_t alpha_value,
                            DualValue &&dual,
                            const DualStep &dual_step,
                            scalar_t alpha_dual) {
    if (value.size() == 0) {
        return;
    }
    value.noalias() += alpha_value * value_step;
    dual.noalias() += alpha_dual * dual_step;
}

template <typename DualValue, typename DualStep>
inline void update_pair_bounds(linesearch_config &cfg,
                               const vector &value,
                               const vector &value_step,
                               const DualValue &dual,
                               const DualStep &dual_step,
                               scalar_t fraction_to_boundary = scalar_t(0.995)) {
    if (value.size() == 0) {
        return;
    }
    cfg.primal.alpha_max = std::min(cfg.primal.alpha_max,
                                    alpha_max(value, value_step, fraction_to_boundary));
    cfg.dual.alpha_max = std::min(cfg.dual.alpha_max,
                                  alpha_max(dual, dual_step, fraction_to_boundary));
}

} // namespace moto::solver::positivity
