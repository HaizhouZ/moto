#ifndef MOTO_OCP_LIFTED_HPP
#define MOTO_OCP_LIFTED_HPP

#include <moto/ocp/constr.hpp>
#include <moto/core/linear_backend.hpp>
#include <moto/core/sparse_matrix.hpp>
#include <functional>
#include <mutex>
#include <unordered_map>

namespace moto {

class generic_dynamics;
using generic_lifted = generic_dynamics;
class ocp_base;
struct lag_data;
using dynamics = utils::shared<generic_dynamics>;
using lifted = dynamics;

/// Meta-only block system supplied to the user's elimination-graph builder.
/// Every block is present with its true dimensions. A structurally absent
/// Jacobian is represented by a shaped MX zero, so the graph may add a
/// regularization term to it without special casing.
struct lifted_symbolic_partition {
  std::string name;
  size_t uid = 0;
  size_t source_uid = 0;
  field_t field = __undefined;
  size_t offset = 0;
  size_t size = 0;
};

struct lifted_symbolic_system;
struct lifted_symbolic_intermediate;
struct lifted_symbolic_projection;

/// Reusable symbolic solve handle. The matrix remains an ordinary MX value;
/// the MX translator recognizes repeated solves and emits one shared runtime
/// factorization for that value.
struct lifted_symbolic_factor {
  cs::MX matrix;
  cs::MX inverse;

  cs::MX solve(const cs::MX &rhs) const;
};

/// A named Jacobian block used while authoring an elimination graph.
/// Parameters created here are ordinary __p symbols; the block supplies a
/// stable default name and maps the symbol into the MX graph.
struct lifted_symbolic_block {
  cs::MX value;
  std::string parameter_name;
  const lifted_symbolic_system *system = nullptr;

  var param(sym::default_val_t default_value = sym::default_val_none_t(),
            std::string name = {}, size_t dim = 1) const;
  lifted_symbolic_block rows(size_t begin, size_t end) const;
  cs::MX add_diag(const sym &parameter) const;
};

struct lifted_symbolic_system {
  cs::MX dyn_y, dyn_l, lift_y, lift_l;
  cs::MX dyn_x, dyn_u, lift_x, lift_u;
  cs::MX dyn_residual, lift_residual;
  cs::MX action_rhs;
  std::vector<lifted_symbolic_partition> equations;
  std::vector<lifted_symbolic_partition> variables;

  lifted_symbolic_block block(field_t equation, field_t variable) const;
  lifted_symbolic_block block(std::string_view equation,
                              std::string_view variable) const;
  lifted_symbolic_block jac(const generic_func &equation,
                            const sym &variable) const;
  lifted_symbolic_block jac(const cs::SX &equation,
                            const sym &variable) const;
  cs::MX residual(const generic_func &equation) const;
  cs::MX residual(std::string_view equation) const;
  cs::MX h_l() const;
  cs::MX h_x() const;
  cs::MX h_u() const;
  cs::MX h() const;
  lifted_symbolic_factor solve(const cs::MX &matrix, bool spd = false) const;
  lifted_symbolic_projection eliminate(
      const std::function<cs::MX(const cs::MX &)> &solve,
      std::vector<lifted_symbolic_intermediate> intermediates = {}) const;

  // Internal hooks installed by generic_dynamics while the builder runs.
  mutable std::function<var(std::string, size_t, sym::default_val_t)>
      parameter_factory_;
  mutable std::function<cs::MX(const sym &)> parameter_resolver_;
  mutable std::function<lifted_symbolic_block(const cs::SX &, const sym &)>
      jacobian_factory_;
  mutable std::unordered_map<size_t, cs::MX> parameter_by_uid_;
  mutable std::shared_ptr<std::vector<cs::MX>> spd_factors_ =
      std::make_shared<std::vector<cs::MX>>();
  var make_parameter(std::string name, size_t dim,
                     sym::default_val_t default_value) const;
  cs::MX resolve_parameter(const sym &parameter) const;
  cs::MX block_value(field_t equation, field_t variable) const;
  friend struct lifted_symbolic_block;
  friend class generic_dynamics;
};

struct lifted_symbolic_intermediate {
  std::string name;
  cs::MX value;
};

/// Result of a user-authored elimination graph. Finalization uses these
/// expressions both to infer the projected panel layouts and to compile the
/// runtime panel kernel for h_l^{-1}[h_x,h_u,h].
struct lifted_symbolic_projection {
  cs::MX response_x;
  cs::MX response_u;
  cs::MX response_residual;
  std::vector<lifted_symbolic_intermediate> intermediates;
  cs::MX response_action;
};

using lifted_elimination_builder =
    std::function<lifted_symbolic_projection(const lifted_symbolic_system &)>;

/// Unified dynamics/lifting group. Ordinary dynamics eliminate their predicted
/// state arguments. A generalized group additionally owns explicit equality
/// sub-constraints and an elimination graph for the coupled [y; l] system.
class generic_dynamics : public generic_constr {
  public:
    using base = generic_constr;
    struct approx_data : public generic_constr::approx_data {
      lag_data::approx_data *approx_ = nullptr;
      lag_data::dynamics_data *dyn_proj_ = nullptr;
      vector_ref proj_f_res_;
      aligned_map_t f_x_{nullptr, 0, 0};
      aligned_map_t f_u_exclusive_{nullptr, 0, 0};
      aligned_map_t proj_f_u_exclusive_{nullptr, 0, 0};
      std::vector<aligned_map_t> f_u_shared_, proj_f_u_shared_;
      aligned_map_t proj_f_x_{nullptr, 0, 0};
      sparse_matrix proj_l_x_, proj_l_u_;
      vector proj_l_res_;
      std::unordered_map<std::string, sparse_matrix> lifted_intermediates_;

      explicit approx_data(generic_constr::approx_data &&rhs);
      approx_data(generic_constr::approx_data &&rhs, bool sparse_projection,
                  bool sparse_raw_jacobian = false);
    };
    using base::base;

    struct projection_panel {
      size_t argument = 0;
      sp_info block;
    };

    /// Mark the primal arguments recovered by this lifting group. Patterns are
    /// inferred from the finalized function; callers never provide sparsity.
    void mark_lifted(const var_inarg_list &args);
    bool is_lifted(const sym &arg) const;
    const var_list &lifted_args() const { return lifted_args_; }
    size_t lifted_tdim() const;

    /// Add an equality block to this dynamics group's coupled elimination
    /// system. The dependency is graph-owned, so adding the dynamics to a
    /// stage also adds its sub-constraints exactly once.
    void add_subconstraint(const constr &constraint);
    const std::vector<constr> &subconstraints() const {
      return subconstraints_;
    }
    bool owns_subconstraint(const generic_constr &constraint) const;

    void mark_shared_inputs(const var_inarg_list &args);
    bool input_shared(const sym &s) const;

    /// Return an ordinary generic_lifted expression whose dynamics interfaces
    /// are implemented by the supplied MX elimination graph. The source
    /// expression is not mutated. Parameters declared by symbolic blocks are
    /// ordinary __p dependencies of the returned expression.
    lifted set_elimination_graph(lifted_elimination_builder builder) const;
    lifted with_elimination_graph(
        lifted_elimination_builder builder,
        const std::vector<constr> &subconstraints = {}) const;
    bool has_elimination_graph() const {
      return static_cast<bool>(elimination_builder_);
    }
    const var_list &elimination_parameters() const {
      return elimination_parameter_state_->parameters;
    }
    lifted_symbolic_projection derive_elimination_graph(
        const lifted_symbolic_system &system) const;
    virtual size_t elimination_source_uid() const { return uid(); }

    virtual void compute_project_jacobians(func_approx_data &data) const;
    virtual void compute_project_residual(func_approx_data &data) const;
    virtual void compute_project_derivatives(func_approx_data &data) const {
      compute_project_jacobians(data);
      compute_project_residual(data);
    }
    virtual void apply_lifted_jacobian_inverse_transpose(
        func_approx_data &data, vector_ref v, vector_ref dst) const;
    void apply_jac_y_inverse_transpose(func_approx_data &data,
                                       vector_ref v,
                                       vector_ref dst) const {
      apply_lifted_jacobian_inverse_transpose(data, v, dst);
    }
    /// A stage with explicit __l variables delegates the coupled [y;l]
    /// projection and actions to exactly one lifted group.
    virtual bool owns_stage_elimination() const {
      return !subconstraints_.empty();
    }
    /// Generic forward/transpose action used by refinement and dual recovery.
    virtual void solve_stage_lifted_system(
        func_approx_data &, const matrix &, matrix &, bool) const;
    virtual std::span<const projection_panel> projected_panel_sparsity() const {
      return {};
    }
    /// Function-local panels of the unprojected constraint Jacobian. Most
    /// constraints use jac_sparsity(); structured dynamics may emit several
    /// panels per argument and expose that exact layout here.
    virtual std::span<const projection_panel> jacobian_panel_sparsity() const {
      return {};
    }
  protected:
    struct elimination_parameter_state {
      mutable std::mutex mutex;
      var_list parameters;
    };

    var_list lifted_args_;
    std::set<size_t> lifted_arg_uids_;
    std::vector<constr> subconstraints_;
    var_list shared_inputs_;
    std::set<size_t> shared_inputs_indices_;
    mutable std::shared_ptr<elimination_parameter_state>
        elimination_parameter_state_ =
            std::make_shared<elimination_parameter_state>();
    lifted_elimination_builder elimination_builder_;

    var get_or_create_elimination_parameter(
        std::string name, size_t dim,
        sym::default_val_t default_value) const;
    void install_elimination_graph(lifted_elimination_builder builder);
    cs::MX use_elimination_parameter(const sym &parameter) const;
    void set_lifted_arguments(const var_inarg_list &args);
    void substitute(const sym &arg, const sym &rhs) override;
    void finalize_impl() override;
    virtual void prepare_dynamics_codegen() {}
};

/// CasADi-backed lifted equality group. Its residual rows are paired with the
/// supplied lifted variables; the stage lifting operator owns elimination.
class implicit_lifted : public generic_dynamics {
  public:
    struct approx_data : public generic_constr::approx_data {
      explicit approx_data(generic_constr::approx_data &&rhs)
          : generic_constr::approx_data(std::move(rhs)) {}
    };

    implicit_lifted(const std::string &name, const cs::SX &out,
                    const var_inarg_list &lifted_args,
                    approx_order order = approx_order::second);
    static lifted create(const std::string &name, const cs::SX &out,
                         const var_inarg_list &lifted_args,
                         approx_order order = approx_order::second);

    void compute_project_jacobians(func_approx_data &) const override {}
    void compute_project_residual(func_approx_data &) const override {}
    void apply_lifted_jacobian_inverse_transpose(
        func_approx_data &, vector_ref, vector_ref) const override {
        throw std::logic_error(
            "implicit_lifted inverse action belongs to the stage lifting operator");
    }
    func_approx_data_ptr_t create_approx_data(
        sym_data &primal, lag_data &raw, shared_data &shared) const override {
      return func_approx_data_ptr_t(
          make_approx<implicit_lifted>(primal, raw, shared));
    }
  protected:
    clone_ptr clone() const override { return new implicit_lifted(*this); }
};

} // namespace moto

#endif // MOTO_OCP_LIFTED_HPP
