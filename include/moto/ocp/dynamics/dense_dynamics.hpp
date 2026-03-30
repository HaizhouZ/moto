#ifndef MOTO_OCP_DENSE_DYNAMICS_HPP
#define MOTO_OCP_DENSE_DYNAMICS_HPP

#include <moto/ocp/dynamics.hpp>
#include <moto/utils/movable_ptr.hpp>

namespace moto {

/// fwd declaration
namespace utils {
struct blasfeo_lu;
}

/**
 * @brief basic dense dynamics implementation
 * @note it requires the state variables being clustered together in the state vector
 * and will the the first arg in its arg list to compute the indices
 */
class dense_dynamics : public generic_dynamics {
  public:
    using base = generic_dynamics;
    struct approx_data : public generic_dynamics::approx_data {
        // sparse_mat proj_f_x_;
        // sparse_mat proj_f_u_;
        using lu_t = utils::blasfeo_lu;
        movable_ptr<lu_t> lu_;                             ///< LU decomposition for dense dynamics
        aligned_map_t f_x_, f_y_;                          ///< Jacobian of f_y
        aligned_map_t f_u_exclusive_, proj_f_u_exclusive_; ///< Jacobian of f_u (exclusive of shared inputs)
        std::vector<aligned_map_t> f_u_shared_;            ///< Jacobian of other fields
        aligned_map_t proj_f_x_;                           ///< Jacobian of x
        std::vector<aligned_map_t> proj_f_u_shared_;       ///< projection of f_u
        approx_data(generic_constr::approx_data &&rhs);
        void reset() override;
        ~approx_data();
    };

    using base::base;

    func_approx_data_ptr_t create_approx_data(sym_data &primal,
                                              merit_data &raw,
                                              shared_data &shared) const override {
        return func_approx_data_ptr_t(make_approx<dense_dynamics>(primal, raw, shared));
    }

    void compute_project_jacobians(func_approx_data &data) const override;
    void compute_project_residual(func_approx_data &data) const override;
    void apply_jac_y_inverse_transpose(func_approx_data &data, vector &v, vector &dst) const override;

    /// @brief mark the shared inputs in the dynamics
    /// @note should be called before finalization
    void mark_shared_inputs(const var_inarg_list &args);

    bool input_shared(const sym &s) const {
        return shared_inputs_indices_.contains(s.uid());
    } ///< check if an input variable is shared

    size_t active_dim_exclusive_inputs(const ocp *prob) const;
    size_t active_dim_shared_inputs(const ocp *prob) const;
    size_t active_num_exclusive_inputs(const ocp *prob) const;
    size_t active_num_shared_inputs(const ocp *prob) const;

  private:
    struct info : public generic_func::info {
        size_t num_exclusive_inputs_ = 0; ///< number of exclusive input variables (i.e., not shared)
        size_t num_shared_inputs_ = 0;    ///< number of shared input variables
        size_t dim_exclusive_inputs_ = 0; ///< dimension of exclusive input variables
        size_t dim_shared_inputs_ = 0;    ///< dimension of shared input variables
        using generic_func::info::info;
        /// @brief move constructor, leave the reset to default after move
        info(generic_func::info &&rhs) : generic_func::info(std::move(rhs)) {}
        /// @note no clone for now
    };
    info &get_info() const { return (info &)(info_); } ///< get info
    var_list shared_inputs_;
    std::set<size_t> shared_inputs_indices_; ///< list of shared input variable uids
    /// @brief handles shared inputs during finalization
    /// @details it will compute the num and dim of exclusive/shared inputs
    void finalize_impl() override;
    /// @brief override substitute to handle shared inputs
    void substitute(const sym &arg, const sym &rhs) override;
    /// @brief setup ocpwise info, will compute the num and dim of exclusive/shared inputs for the problem
    bool setup_ocpwise_info(const ocp *prob) const override;
};
} // namespace moto

#endif