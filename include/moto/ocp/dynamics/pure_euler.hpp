#ifndef MOTO_OCP_PURE_EULER_HPP
#define MOTO_OCP_PURE_EULER_HPP

#include <moto/ocp/dynamics/semi_implicit_euler.hpp>

namespace moto {

/// Complete first-order kinematic dynamics with a structured configuration inverse.
class pure_euler : public semi_implicit_euler {
public:
  using base = semi_implicit_euler;
  using base::base;

protected:
  clone_ptr clone() const override { return new pure_euler(*this); }
  cs::SX symbolic_inverse(const cs::SX &fy) const override;
};

} // namespace moto
#endif
