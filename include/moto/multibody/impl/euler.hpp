#ifndef MOTO_MULTIBODY_EULER_HPP
#define MOTO_MULTIBODY_EULER_HPP

#include <moto/multibody/fwd.hpp>
#include <moto/ocp/dynamics.hpp>

namespace moto {
namespace multibody {
struct euler_data;
struct stacked_euler;
// Euler angles utilities
class euler : public generic_func {
  public:
    enum class v_int_type : size_t {
        _explicit = 0,
        _mid_point,
        _implicit,
    };

  private:
    using base = generic_func;
    friend struct stacked_euler;
    var q_, v_, a_, dt_;
    cs::SX pos_integrate_, pos_diff_, pos_step_;
    struct jac_block {
        bool empty = true;
        cs::SX dense, diag;
        struct {
            int r_st, c_st, nrow, ncol;
        } dense_block;
        bool has_block = false;
        bool has_diag = false; /// @note assumption diag exists if has_block is true
        void setup(const cs::SX &mat, size_t c_offset = 0);
    };
    struct {
        jac_block dqn, dq, dvn, dv;
    } pos_diff_jac_;

    struct {
        jac_block dq, dv, da;
    } pos_diff_proj_jac_;

    size_t n_jac_output_ = 0;

    using binary_jac_t = std::array<cs::SX, 2>;
    binary_jac_t dpos_diff_, dpos_int_;

    v_int_type v_int_type_ = v_int_type::_explicit;

    friend class func;
    using wrapper_type = func;

  public:
    const auto &pos() const { return q_; }
    const auto &vel() const { return v_; }
    const auto &acc() const { return a_; }
    const auto &dt() const { return dt_; }
    static func from_urdf(const std::string &urdf_path,
                          var dt,
                          root_joint_t root_joint = root_joint_t::xyz_quat,
                          euler::v_int_type v_int = euler::v_int_type::_implicit);
    euler() = default;
    euler(const std::string &name,
          const var &q, const var &v, const var &a, const var &dt, cs::SX pos_step,
          cs::SX pos_diff, binary_jac_t dpos_diff, cs::SX pos_int, binary_jac_t dpos_int);
    void load_external_impl(const std::string &path) override;
    void finalize_impl() override;
    void setup_data(euler_data &data) const;
    void jacobian_impl(func_approx_data &data) const override;
    void compute_project_derivatives(euler_data &data) const;

    /// @note assign new syms for the q, v, a
    func share(const std::string &name = "") const;
};

} // namespace multibody
} // namespace moto

#endif // MOTO_MULTIBODY_EULER_HPP