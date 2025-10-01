#include <moto/multibody/quaternion.hpp>

namespace moto {
namespace multibody {
/// @brief quaternion multiplication
cs::SX quaternion::multiply(const cs::SX &q0, const cs::SX &q1) {
    return cs::SX::vertcat(std::vector{
        q0(3) * q1(0) + q0(0) * q1(3) + q0(1) * q1(2) - q0(2) * q1(1),
        q0(3) * q1(1) - q0(0) * q1(2) + q0(1) * q1(3) + q0(2) * q1(0),
        q0(3) * q1(2) + q0(0) * q1(1) - q0(1) * q1(0) + q0(2) * q1(3),
        q0(3) * q1(3) - q0(0) * q1(0) - q0(1) * q1(1) - q0(2) * q1(2),
    });
}
/// @brief quaterion inverse
cs::SX quaternion::inverse(const cs::SX &q) {
    return cs::SX::vertcat(std::vector{-q(0), -q(1), -q(2), q(3)});
}
/// @brief identity quaternion [0, 0, 0, 1]
cs::SX quaternion::symbolic_identity() {
    static auto id = cs::SX::vertcat(std::vector{cs::SX::zeros(3), cs::SX(1.)});
    return id;
}
/// exponential map from R^3 to unit quaternion
cs::SX quaternion::exp3(const cs::SX &dq, scalar_t tolerance) {
    auto theta = cs::SX::norm_2(dq);
    auto axis = dq / theta;
    auto w = cs::SX::cos(theta / 2);
    auto xyz = axis * cs::SX::sin(theta / 2);
    return cs::SX::if_else(
        theta < tolerance,
        symbolic_identity(),
        cs::SX::vertcat(std::vector{xyz, w}) // normal case
    );
}
/// logarithm map from unit quaternion to R^3
cs::SX quaternion::log3(const cs::SX &q, scalar_t tolerance) {
    auto w = q(3);
    auto xyz = q(cs::Slice(0, 3));
    auto sin_half_theta = cs::SX::norm_2(xyz);
    auto cos_half_theta = w;
    auto half_theta = cs::SX::atan2(sin_half_theta, cos_half_theta);
    auto axis = xyz / sin_half_theta;
    return cs::SX::if_else(
        sin_half_theta < tolerance,
        cs::SX::zeros(3),
        axis * (2 * half_theta) // normal case
    );
}
/// @brief quaternion integration using exponential map
cs::SX quaternion::symbolic_integrate(const cs::SX &q, const cs::SX &dq) const {
    auto q_next = multiply(q, exp3(dq));
    return q_next / cs::SX::norm_2(q_next); // normalize to unit quaternion
}
/// @brief quaternion difference using logarithm map q1 \ominus q0 = log(q0^{-1} * q1)
cs::SX quaternion::symbolic_difference(const cs::SX &q1, const cs::SX &q0) const {
    auto dq = multiply(inverse(q0), q1);
    return log3(dq);
}

} // namespace multibody
} // namespace moto