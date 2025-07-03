#ifndef MOTO_SOLVER_SOLVER_DATA_HPP
#define MOTO_SOLVER_SOLVER_DATA_HPP

#include <moto/ocp/approx_storage.hpp>
#include <moto/ocp/problem.hpp>

namespace moto {
namespace solver {
constexpr field_t primal_fields[] = {__x, __u, __y};

struct solver_data {
    size_t nx, nu;
    // value function
    row_vector &Q_x;
    row_vector &Q_u;
    row_vector &Q_y;
    matrix &Q_xx;
    matrix &Q_ux;
    matrix &Q_uu;
    matrix &Q_yx;
    matrix &Q_yy;
    // linear rollout
    array<vector, std::size(primal_fields)> prim_rollout_;
    /// @brief create solver data
    /// @param prob ocp to initialize nx, nu
    /// @param dense_ to initialize Q_ ref and rollout_ primal data
    solver_data(const ocp_ptr_t &prob, approx_storage *dense_);
};

} // namespace solver
} // namespace moto

#endif // MOTO_SOLVER_SOLVER_DATA_HPP