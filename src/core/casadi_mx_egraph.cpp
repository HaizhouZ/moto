#include "casadi_mx_egraph.hpp"

#include <moto/core/linear_egraph.hpp>

#include <cstdlib>
#include <iostream>
#include <unordered_map>
#include <unordered_set>

namespace moto::linear_backend::detail {
namespace {

bool is_product(const casadi::MX &value) {
  return value.op() == casadi::OP_MTIMES && value.n_dep() == 3 &&
         value.dep(0).is_zero();
}

bool is_algebraic(const casadi::MX &value) {
  return is_product(value) || value.op() == casadi::OP_TRANSPOSE;
}

sparse_pattern pattern(const casadi::Sparsity &value) {
  sparse_pattern result{static_cast<size_t>(value.size1()),
                        static_cast<size_t>(value.size2())};
  for (const auto index : value.get_colind())
    result.colind.push_back(static_cast<size_t>(index));
  for (const auto index : value.get_row())
    result.row.push_back(static_cast<size_t>(index));
  return result;
}

struct importer {
  const std::unordered_map<const casadi::MXNode *, matrix_layout> &input_layout;
  linear_egraph graph;
  std::unordered_map<const casadi::MXNode *, linear_eclass_id> imported;
  std::unordered_map<const casadi::MXNode *, size_t> leaf_by_node;
  std::vector<casadi::MX> leaves;

  linear_eclass_id add(const casadi::MX &value) {
    if (const auto found = imported.find(value.get()); found != imported.end())
      return found->second;
    linear_eclass_id id;
    if (is_product(value)) {
      id = graph.add_multiply(add(value.dep(1)), add(value.dep(2)));
    } else if (value.op() == casadi::OP_TRANSPOSE) {
      id = graph.add_transpose(add(value.dep(0)));
    } else if (value.is_eye()) {
      id = graph.add_identity(value.size1());
    } else {
      size_t leaf;
      if (const auto found = leaf_by_node.find(value.get());
          found != leaf_by_node.end()) {
        leaf = found->second;
      } else {
        leaf = leaves.size();
        leaves.push_back(value);
        leaf_by_node.emplace(value.get(), leaf);
      }
      const sparse_pattern exact = pattern(value.sparsity());
      const auto found = input_layout.find(value.get());
      matrix_layout layout = found == input_layout.end()
                                 ? panelize_pattern(exact)
                                 : found->second;
      id = graph.add_leaf(leaf, std::move(layout), exact);
    }
    imported.emplace(value.get(), id);
    return id;
  }

  casadi::MX extract(const casadi::MX &root) {
    const auto root_id = add(root);
    const auto original = graph.extract(root_id).at(root_id).cost;
    const auto saturation = graph.saturate();
    const auto extraction = graph.extract(root_id);
    if (std::getenv("MOTO_TRACE_EGRAPH") &&
        extraction.at(extraction.root).cost < original) {
      const auto &optimized = extraction.at(extraction.root).cost;
      std::cerr << "linear e-graph " << root.size1() << 'x' << root.size2()
                << " nodes=" << saturation.nodes
                << " scalar_products=" << original.scalar_products << "->"
                << optimized.scalar_products << " panel_products="
                << original.panel_products << "->" << optimized.panel_products
                << " temporaries=" << original.temporary_scalars << "->"
                << optimized.temporary_scalars << '\n';
    }
    std::unordered_map<linear_eclass_id, casadi::MX> values;
    const auto build = [&](const auto &self, linear_eclass_id id) -> casadi::MX {
      if (const auto found = values.find(id); found != values.end())
        return found->second;
      const auto &term = extraction.at(id);
      casadi::MX result;
      if (term.op == linear_egraph_op::leaf)
        result = leaves.at(term.leaf);
      else if (term.op == linear_egraph_op::identity)
        result = casadi::MX::eye(term.value.rows());
      else if (term.op == linear_egraph_op::transpose)
        result = self(self, term.children[0]).T();
      else
        result = casadi::MX::mtimes(self(self, term.children[0]),
                                    self(self, term.children[1]));
      values.emplace(id, result);
      return result;
    };
    return build(build, extraction.root);
  }
};

using parent_map =
    std::unordered_map<const casadi::MXNode *, std::vector<casadi::MX>>;

void collect(const casadi::MX &value,
             std::unordered_set<const casadi::MXNode *> &visited,
             parent_map &parents, std::vector<casadi::MX> &algebraic) {
  if (!value.get() || !visited.insert(value.get()).second) return;
  if (is_algebraic(value)) algebraic.push_back(value);
  for (casadi_int i = 0; i < value.n_dep(); ++i) {
    const auto dependency = value.dep(i);
    parents[dependency.get()].push_back(value);
    collect(dependency, visited, parents, algebraic);
  }
}

} // namespace

std::vector<std::vector<casadi::MX>> optimize_casadi_mx_graph(
    const std::vector<casadi::MX> &inputs,
    const std::vector<std::vector<casadi::MX>> &output_entries,
    std::span<const matrix_layout> input_layouts) {
  if (!input_layouts.empty() && input_layouts.size() != inputs.size())
    throw std::invalid_argument("MX e-graph input-layout count mismatch");
  std::unordered_map<const casadi::MXNode *, matrix_layout> layouts;
  for (size_t i = 0; i < inputs.size(); ++i) {
    const sparse_pattern exact = pattern(inputs[i].sparsity());
    const bool supplied = !input_layouts.empty() &&
                          (input_layouts[i].rows || input_layouts[i].cols);
    layouts.emplace(inputs[i].get(), supplied ? input_layouts[i]
                                              : panelize_pattern(exact));
  }

  parent_map parents;
  std::vector<casadi::MX> algebraic;
  std::unordered_set<const casadi::MXNode *> visited;
  for (const auto &entry : output_entries)
    for (const auto &output : entry)
      collect(output, visited, parents, algebraic);

  std::vector<casadi::MX> targets;
  for (const auto &value : algebraic) {
    const auto found = parents.find(value.get());
    if (found == parents.end() ||
        std::ranges::any_of(found->second, [](const casadi::MX &parent) {
          return !is_algebraic(parent);
        }))
      targets.push_back(value);
  }
  if (targets.empty()) return output_entries;

  std::vector<casadi::MX> replacements;
  replacements.reserve(targets.size());
  for (const auto &target : targets) {
    importer optimizer{layouts};
    replacements.push_back(optimizer.extract(target));
  }
  auto result = output_entries;
  for (auto &entry : result)
    entry = casadi::MX::graph_substitute(entry, targets, replacements);
  return result;
}

} // namespace moto::linear_backend::detail
