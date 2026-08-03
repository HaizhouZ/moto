#ifndef MOTO_MULTIBODY_CASADI_MANIFOLD_HPP
#define MOTO_MULTIBODY_CASADI_MANIFOLD_HPP

#include <moto/ocp/sym.hpp>

namespace moto::multibody {

class casadi_manifold final : public sym {
    cs::Function integrate_fn_;
    cs::Function difference_fn_;

    casadi_manifold(const std::string &name, size_t dim, size_t tdim,
                    const cs::SX &q, const cs::SX &dq,
                    const cs::SX &integrated, const cs::SX &other,
                    const cs::SX &difference, field_t field,
                    default_val_t default_val);

  public:
    cs::SX symbolic_integrate(const cs::SX &q,
                              const cs::SX &dq) const override;
    cs::SX symbolic_difference(const cs::SX &q1,
                               const cs::SX &q0) const override;

    static std::pair<var, var> create(
        const std::string &name, const cs::SX &q, const cs::SX &dq,
        const cs::SX &integrated, const cs::SX &other,
        const cs::SX &difference,
        default_val_t default_val = default_val_none_t());
    var clone(const std::string &name) const override {
        return clone_states<casadi_manifold>(name);
    }

  protected:
    clone_ptr clone() const override { return new casadi_manifold(*this); }
};

} // namespace moto::multibody

#endif
