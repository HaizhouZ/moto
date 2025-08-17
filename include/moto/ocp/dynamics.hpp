#ifndef MOTO_OCP_DYNAMICS_HPP
#define MOTO_OCP_DYNAMICS_HPP

#include <moto/ocp/constr.hpp>
#include <moto/spmm/sparse_mat.hpp>

namespace moto {

/// @brief generic dynamics
class generic_dynamics : public generic_constr {
  public:
    using base = generic_constr;
    struct approx_data : public base::approx_data {
        using aligned_map_t = matrix::AlignedMapType;
        using aligned_vector_map_t = vector::AlignedMapType;
#define NULL_INIT_MAP(name) name(nullptr, 0, 0)
#define NULL_INIT_VECMAP(name) name(nullptr, 0)
        merit_data::approx_data *approx_;
        merit_data::dynamics_data *dyn_proj_;
        vector_ref proj_f_res_; ///< projection of f_res
        approx_data(base::approx_data &&rhs);
    };
    using base::base;
    virtual void compute_project_derivatives(func_approx_data &data) const = 0;
    virtual void apply_jac_y_inverse_transpose(func_approx_data &data, vector_ref v, vector_ref dst) const { dst = v; };
    template <typename T, typename src_type>
    static void setup_map(T &v, src_type &src) {
        new (&v) std::remove_cvref_t<T>(src.data(), src.rows(), src.cols());
    }
};

} // namespace moto

#endif // MOTO_OCP_DYNAMICS_HPP