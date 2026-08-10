#ifndef MOTO_CASADI_MX_EGRAPH_HPP
#define MOTO_CASADI_MX_EGRAPH_HPP

#include <moto/core/linear_backend.hpp>

#include <casadi/casadi.hpp>

namespace moto::linear_backend::detail {

std::vector<std::vector<casadi::MX>> optimize_casadi_mx_graph(
    const std::vector<casadi::MX> &inputs,
    const std::vector<std::vector<casadi::MX>> &output_entries,
    std::span<const matrix_layout> input_layouts);

} // namespace moto::linear_backend::detail

#endif
