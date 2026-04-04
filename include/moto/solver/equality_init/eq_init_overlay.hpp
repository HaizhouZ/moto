#pragma once

#include <moto/ocp/problem.hpp>
#include <moto/solver/soft_constr/pmm_constr.hpp>

namespace moto {
struct node_data;
}

namespace moto::solver::equality_init {

struct equality_init_overlay_settings {
    scalar_t rho_eq = 1.0;
};

class eq_init_pmm_constr final : public pmm_constr {
  public:
    eq_init_pmm_constr(const std::string &name,
                       const constr &source,
                       size_t source_pos,
                       scalar_t rho);

    void value_impl(func_approx_data &data) const override;
    void jacobian_impl(func_approx_data &data) const override;
    void hessian_impl(func_approx_data &data) const override;

    const constr &source() const { return source_; }
    field_t source_field() const { return source_->field(); }
    size_t source_pos() const { return source_pos_; }

    DEF_DEFAULT_CLONE(eq_init_pmm_constr)

  private:
    constr source_;
    size_t source_pos_ = 0;
};

ocp_ptr_t build_equality_init_overlay_problem(const ocp_ptr_t &source_prob,
                                              const equality_init_overlay_settings &settings);

void sync_equality_init_overlay_primal(node_data &outer, node_data &overlay);
void sync_equality_init_overlay_duals(node_data &outer, node_data &overlay);
void commit_equality_init_overlay_duals(node_data &outer, node_data &overlay);

} // namespace moto::solver::equality_init
