#ifndef MOTO_CASADI_MX_GRAPH_TRANSLATOR_HPP
#define MOTO_CASADI_MX_GRAPH_TRANSLATOR_HPP

#include <moto/core/linear_backend.hpp>

#include <casadi/casadi.hpp>

#include <filesystem>
#include <memory>
#include <string>

namespace moto::linear_backend::detail {

struct casadi_mx_graph_plan;

void *compile_casadi_mx_graph_source(const std::string &source,
                                     const std::filesystem::path &cache_dir);

struct casadi_mx_graph_instance {
  std::shared_ptr<const casadi_mx_graph_plan> plan;
  std::shared_ptr<std::vector<sparse_matrix>> owned_workspace;
  std::vector<sparse_matrix> *workspace = nullptr;
  std::vector<size_t> workspace_slots;
  mutable std::vector<std::vector<scalar_t *>> slot_pointers;
  /// Stable flattened pointer table consumed by generated backend programs.
  /// Value-panel slots come first, followed by caller-owned graph pointers.
  mutable std::vector<scalar_t *> backend_pointers;
  mutable std::vector<scalar_t *> bound_external;
  mutable bool workspace_bound = false;
  mutable void *whole_kernel_state = nullptr;

  casadi_mx_graph_instance(std::shared_ptr<const casadi_mx_graph_plan> plan,
                           std::vector<sparse_matrix> *workspace);
  ~casadi_mx_graph_instance();
  void run(size_t entry, std::span<scalar_t *> pointers) const;
};

std::shared_ptr<const casadi_mx_graph_plan> translate_casadi_mx_graph(
    const casadi::Function &function, std::vector<size_t> entry_outputs,
    std::span<const matrix_layout> input_layouts = {},
    const std::filesystem::path &cache_dir = "gen/linear_backend",
    size_t spd_outputs = 0);

size_t casadi_mx_graph_inputs(const casadi_mx_graph_plan &plan);
size_t casadi_mx_graph_outputs(const casadi_mx_graph_plan &plan);
size_t casadi_mx_graph_entries(const casadi_mx_graph_plan &plan);

} // namespace moto::linear_backend::detail

#endif
