#include <moto/multibody/casadi_manifold.hpp>

namespace moto::multibody {

casadi_manifold::casadi_manifold(
    const std::string &name, size_t dim, size_t tdim, const cs::SX &q,
    const cs::SX &dq, const cs::SX &integrated, const cs::SX &other,
    const cs::SX &difference, field_t field, default_val_t default_val)
    : sym(name, dim, field, std::move(default_val)),
      integrate_fn_(name + "_symbolic_integrate", {q, dq}, {integrated}),
      difference_fn_(name + "_symbolic_difference", {other, q}, {difference}) {
    if (q.numel() != static_cast<casadi_int>(dim) ||
        other.numel() != static_cast<casadi_int>(dim) ||
        integrated.numel() != static_cast<casadi_int>(dim) ||
        dq.numel() != static_cast<casadi_int>(tdim) ||
        difference.numel() != static_cast<casadi_int>(tdim))
        throw std::runtime_error("casadi manifold expression dimension mismatch");
    tdim_ = tdim;
}

cs::SX casadi_manifold::symbolic_integrate(const cs::SX &q,
                                           const cs::SX &dq) const {
    return integrate_fn_(std::vector<cs::SX>{q, dq}).front();
}

cs::SX casadi_manifold::symbolic_difference(const cs::SX &q1,
                                            const cs::SX &q0) const {
    return difference_fn_(std::vector<cs::SX>{q1, q0}).front();
}

std::pair<var, var> casadi_manifold::create(
    const std::string &name, const cs::SX &q, const cs::SX &dq,
    const cs::SX &integrated, const cs::SX &other,
    const cs::SX &difference, default_val_t default_val) {
    const auto dim = static_cast<size_t>(q.numel());
    const auto tdim = static_cast<size_t>(dq.numel());
    var x(new casadi_manifold(name, dim, tdim, q, dq, integrated, other,
                              difference, __x, default_val));
    var y(new casadi_manifold(name + next_suffix_, dim, tdim, q, dq,
                              integrated, other, difference, __y,
                              std::move(default_val)));
    setup_states(x, y);
    global_registry::add(x);
    global_registry::add(y);
    return {std::move(x), std::move(y)};
}

} // namespace moto::multibody
