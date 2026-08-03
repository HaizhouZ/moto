#include <moto/ocp/dynamics/pure_euler.hpp>

namespace moto {

cs::SX pure_euler::symbolic_inverse(const cs::SX &fy) const {
  if (fy.rows() != fy.columns())
    throw std::runtime_error(
        fmt::format("pure Euler dynamics {} requires square F_y", name()));
  return cs::SX::sparsify(configuration_inverse(fy));
}

} // namespace moto
