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
        ~approx_data();
    };

    using base::base;
    clone_ptr clone() const override;

    func_approx_data_ptr_t create_approx_data(sym_data &primal,
                                              lag_data &raw,
                                              shared_data &shared) const override {
        return func_approx_data_ptr_t(make_approx<dense_dynamics>(primal, raw, shared));
    }

    void compute_project_jacobians(func_approx_data &data) const override;
    void compute_project_residual(func_approx_data &data) const override;
    void apply_jac_y_inverse_transpose(func_approx_data &data, vector &v, vector &dst) const override;

    /// @brief mark the shared inputs in the dynamics
    /// @note should be called before finalization
    void mark_shared_inputs(const var_inarg_list &args);

  private:
    bool input_shared(const sym &s) const;

    var_list shared_inputs_;
    std::set<size_t> shared_inputs_indices_;
    void finalize_impl() override;
    void substitute(const sym &arg, const sym &rhs) override;
};
} // namespace moto

#endif
