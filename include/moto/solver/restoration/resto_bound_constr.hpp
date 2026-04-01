#pragma once

#include <moto/solver/ipm/ipm_constr.hpp>

namespace moto {
namespace solver {

enum class resto_bound_type : size_t {
    positive = 0,
    negative,
};

/**
 * @brief Solver-owned restoration bound variable.
 *
 * @details This is a thin IPM-backed wrapper used by elastic restoration to
 * model solver-managed positive/negative auxiliary variables. The storage
 * symbol lives in field `__s`, so the variable is shareable across
 * restoration-only constraints, but it is not part of the user-facing primal
 * fields and does not participate in the Riccati state/control rollout.
 */
class resto_bound_constr : public ipm_constr {
  public:
    struct approx_data : public ipm_constr::approx_data {
        vector_ref storage_;
        vector storage_backup_;

        approx_data(ipm_constr::approx_data &&rhs);
    };

    resto_bound_type bound_type = resto_bound_type::positive;

    resto_bound_constr() = default;
    resto_bound_constr(resto_bound_type type, size_t dim, const std::string &name = "resto_bound");

    func_approx_data_ptr_t create_approx_data(sym_data &primal, lag_data &raw, shared_data &shared) const override {
        std::unique_ptr<ipm_constr::approx_data> base_d(make_approx<ipm_constr>(primal, raw, shared));
        return func_approx_data_ptr_t(new approx_data(std::move(*base_d)));
    }

    void initialize(data_map_t &data) const override;
    void backup_trial_state(data_map_t &data) const override;
    void restore_trial_state(data_map_t &data) const override;
    void apply_affine_step(data_map_t &data, workspace_data *cfg) const override;

    DEF_DEFAULT_CLONE(resto_bound_constr)
};

} // namespace solver
using resto_bound_constr = solver::resto_bound_constr;
using resto_bound_type = solver::resto_bound_type;
} // namespace moto
