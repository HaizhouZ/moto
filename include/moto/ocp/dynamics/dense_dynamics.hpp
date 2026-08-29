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
        // sparse_matrix proj_f_x_;
        // sparse_matrix proj_f_u_;
        using lu_t = utils::blasfeo_lu;
        movable_ptr<lu_t> lu_;                             ///< LU decomposition for dense dynamics
        aligned_map_t f_y_;
        approx_data(generic_constr::approx_data &&rhs);
        ~approx_data();
    };

    using base::base;
  protected:
    clone_ptr clone() const override;

  public:
    func_approx_data_ptr_t create_approx_data(sym_data &primal,
                                              lag_data &raw,
                                              shared_data &shared) const override {
        return func_approx_data_ptr_t(make_approx<dense_dynamics>(primal, raw, shared));
    }

    void compute_project_jacobians(func_approx_data &data) const override;
    void compute_project_residual(func_approx_data &data) const override;
    void apply_lifted_jacobian_inverse_transpose(
        func_approx_data &data, vector_ref v, vector_ref dst) const override;

    void finalize_impl() override;
};
} // namespace moto

#endif
