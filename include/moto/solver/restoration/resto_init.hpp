#pragma once

#include <moto/core/fwd.hpp>
#include <cmath>
#include <stdexcept>

namespace moto::solver::restoration {

struct elastic_init_pair {
    scalar_t p = 0.;
    scalar_t n = 0.;
    scalar_t z_p = 0.;
    scalar_t z_n = 0.;
    scalar_t lambda = 0.;
    scalar_t weight = 0.;
};

/**
 * @brief IPOPT-style elastic restoration initialization for one scalar residual.
 *
 * @details For fixed residual c, initialize p,n > 0 from the barrier-smoothed
 * exact-penalty subproblem
 *   min rho (p + n) - mu log p - mu log n
 *   s.t. c - p + n = 0.
 * Then initialize the associated bound duals by z_p = mu / p, z_n = mu / n.
 */
inline elastic_init_pair initialize_elastic_pair(scalar_t c, scalar_t rho, scalar_t mu_bar) {
    if (!(rho > 0.)) {
        throw std::runtime_error("initialize_elastic_pair requires rho > 0");
    }
    if (!(mu_bar > 0.)) {
        throw std::runtime_error("initialize_elastic_pair requires mu_bar > 0");
    }

    const scalar_t disc = (mu_bar - rho * c) * (mu_bar - rho * c) + scalar_t(2.) * rho * mu_bar * c;
    const scalar_t sqrt_disc = std::sqrt(std::max(disc, scalar_t(0.)));
    const scalar_t n = (mu_bar - rho * c + sqrt_disc) / (scalar_t(2.) * rho);
    const scalar_t p = c + n;

    elastic_init_pair out;
    out.p = std::max(p, scalar_t(1e-16));
    out.n = std::max(n, scalar_t(1e-16));
    out.z_p = mu_bar / out.p;
    out.z_n = mu_bar / out.n;
    out.lambda = rho - out.z_p;

    // d lambda / d c for the eliminated smooth elastic potential.
    // With a = mu / p^2 and b = mu / n^2, d lambda / d c = (a b) / (a + b).
    const scalar_t a = out.z_p / out.p;
    const scalar_t b = out.z_n / out.n;
    out.weight = (a > 0. && b > 0.) ? (a * b) / (a + b) : scalar_t(0.);
    return out;
}

} // namespace moto::solver::restoration
