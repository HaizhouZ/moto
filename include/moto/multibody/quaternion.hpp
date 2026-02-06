#ifndef MOTO_MULTIBODY_QUATERNION_HPP
#define MOTO_MULTIBODY_QUATERNION_HPP

#include <moto/ocp/sym.hpp>

namespace moto {
namespace multibody {
/// @brief quaternion symbolic variable [x, y, z, w] with w as the scalar part
struct quaternion : public sym {
  public:
    static auto &identity() {
        static vector id = (vector(4) << 0.0, 0.0, 0.0, 1.0).finished();
        return id;
    }

  private:
    quaternion(const std::string &name, field_t field = __x)
        : sym(name, 4, field, identity()) {
        tdim_ = 3; // tangent space dimension
    }

  public:
    /// @brief quaternion multiplication
    static cs::SX multiply(const cs::SX &q0, const cs::SX &q1);
    /// @brief quaterion inverse
    static cs::SX inverse(const cs::SX &q);
    /// @brief identity quaternion [0, 0, 0, 1]
    static cs::SX symbolic_identity();
    /// exponential map from R^3 to unit quaternion
    static cs::SX exp3(const cs::SX &w, scalar_t tolerance = 1e-12);
    /// logarithm map from unit quaternion to R^3
    static cs::SX log3(const cs::SX &q, scalar_t tolerance = 1e-12);
    /// @brief quaternion integration using exponential map
    cs::SX symbolic_integrate(const cs::SX &q, const cs::SX &dq) const override final;
    /// @brief quaternion difference using logarithm map q1 \ominus q0 = log(q0^{-1} * q1)
    cs::SX symbolic_difference(const cs::SX &q1, const cs::SX &q0) const override final;
    /// clone the quaternion symbolic variable
    var clone(const std::string &name) const override final { return clone_states<quaternion>(name); }
    /// create a new quaternion symbolic variable
    static std::pair<var, var> create(const std::string &name) {
        static var base, base_n;
        if (!bool(base)) {
            // for all quats to share the same int/diff implementation
            base.reset(new quaternion("quat_base", __x));
            base_n.reset(new quaternion(base->get_next_name(), __y));
            setup_states(base, base_n);
            base_n->finalize(true);
        }
        // create a new instance
        auto tmp = base->clone(name);
        return {tmp, tmp->dual()};
    }
};
} // namespace multibody
} // namespace moto

#endif // MOTO_MULTIBODY_QUATERNION_HPP