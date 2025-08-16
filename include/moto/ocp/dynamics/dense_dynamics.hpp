#ifndef MOTO_OCP_DENSE_DYNAMICS_HPP
#define MOTO_OCP_DENSE_DYNAMICS_HPP

#include <moto/ocp/dynamics.hpp>
#include <moto/ocp/problem.hpp>
#include <moto/utils/movable_ptr.hpp>

namespace moto {

/// fwd declaration
template <typename _MatrixType>
class PartialPivLU;

struct dense_dynamics : public func {
    class impl : public generic_dynamics {
      public:
        using base = generic_dynamics;
        struct approx_data : public generic_dynamics::approx_data {
            // sparse_mat proj_f_x_;
            // sparse_mat proj_f_u_;
            using lu_t = Eigen::PartialPivLU<matrix>;
            movable_ptr<lu_t> lu_;                ///< LU decomposition for dense dynamics
            aligned_map_t f_x_, f_y_;             ///< Jacobian of f_y
            std::vector<aligned_map_t> f_u_;      ///< Jacobian of other fields
            aligned_map_t proj_f_x_;              ///< Jacobian of x
            std::vector<aligned_map_t> proj_f_u_; ///< projection of f_u
            approx_data(generic_constr::approx_data &&rhs);
            ~approx_data();
        };

        using base::base;

        friend class dense_dynamics;

        func_approx_data_ptr_t create_approx_data(sym_data &primal,
                                                  merit_data &raw,
                                                  shared_data &shared) const override {
            return func_approx_data_ptr_t(make_approx<impl>(primal, raw, shared));
        }

        void compute_project_derivatives(func_approx_data &data) const override;
    };

    dense_dynamics(const std::string &name, approx_order order, size_t dim = dim_tbd);
    dense_dynamics(const std::string &name, const var_inarg_list &in_args, const cs::SX &out,
                   approx_order order = approx_order::first);
                   
    impl *operator->() const {
        return static_cast<impl *>(func::operator->());
    } ///< convert to impl type
};
} // namespace moto

#endif