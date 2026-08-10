#include <moto/core/linear_egraph.hpp>

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

namespace moto::linear_backend {
namespace {

struct enode {
  linear_egraph_op op = linear_egraph_op::leaf;
  size_t payload = 0;
  std::array<linear_eclass_id, 2> children{};
  size_t arity = 0;

  bool operator==(const enode &) const = default;
};

struct enode_hash {
  size_t operator()(const enode &node) const noexcept {
    size_t seed = static_cast<size_t>(node.op) * size_t{0x9e3779b9} +
                  node.payload;
    for (size_t i = 0; i < node.arity; ++i)
      seed ^= node.children[i] + size_t{0x9e3779b9} + (seed << 6) +
              (seed >> 2);
    return seed;
  }
};

size_t storage_size(const matrix_layout &layout) {
  return std::accumulate(
      layout.panels.begin(), layout.panels.end(), size_t{0},
      [](size_t total, const panel_layout &panel) {
        if (panel.pattern == sparsity::eye) return total;
        return total + (panel.pattern == sparsity::dense
                            ? panel.rows * panel.cols
                            : panel.rows);
      });
}

linear_egraph_cost add_cost(linear_egraph_cost lhs,
                            const linear_egraph_cost &rhs) {
  lhs.scalar_products += rhs.scalar_products;
  lhs.panel_products += rhs.panel_products;
  lhs.temporary_scalars += rhs.temporary_scalars;
  lhs.operations += rhs.operations;
  return lhs;
}

sparse_pattern identity_pattern(size_t dimension) {
  sparse_pattern result{dimension, dimension, {0}, {}};
  for (size_t i = 0; i < dimension; ++i) {
    result.row.push_back(i);
    result.colind.push_back(i + 1);
  }
  return result;
}

matrix_layout identity_layout(size_t dimension) {
  return {dimension,
          dimension,
          {{sparsity::eye, 0, 0, dimension, dimension}}};
}

} // namespace

bool linear_egraph_cost::operator<(const linear_egraph_cost &other) const {
  return std::tie(scalar_products, panel_products, temporary_scalars,
                  operations) <
         std::tie(other.scalar_products, other.panel_products,
                  other.temporary_scalars, other.operations);
}

const linear_extracted_term &linear_extraction::at(linear_eclass_id id) const {
  if (id >= terms.size() || !terms[id])
    throw std::out_of_range("missing extracted linear e-class");
  return *terms[id];
}

struct linear_egraph::impl {
  struct eclass {
    linear_eclass_id parent = 0;
    size_t rank = 0;
    sparse_pattern pattern;
    std::vector<enode> nodes;
  };

  std::vector<eclass> classes;
  std::unordered_map<enode, linear_eclass_id, enode_hash> memo;
  std::unordered_map<size_t, spmm_operand> leaves;
  size_t revision = 0;

  linear_eclass_id find(linear_eclass_id id) {
    if (id >= classes.size()) throw std::out_of_range("invalid linear e-class");
    if (classes[id].parent != id)
      classes[id].parent = find(classes[id].parent);
    return classes[id].parent;
  }

  linear_eclass_id find_const(linear_eclass_id id) const {
    if (id >= classes.size()) throw std::out_of_range("invalid linear e-class");
    while (classes[id].parent != id) id = classes[id].parent;
    return id;
  }

  enode canonical(enode node) {
    for (size_t i = 0; i < node.arity; ++i)
      node.children[i] = find(node.children[i]);
    return node;
  }

  sparse_pattern analyze(const enode &node) {
    if (node.op == linear_egraph_op::leaf)
      return leaves.at(node.payload).pattern;
    if (node.op == linear_egraph_op::identity)
      return identity_pattern(node.payload);
    const auto &lhs = classes[find(node.children[0])].pattern;
    if (node.op == linear_egraph_op::transpose)
      return transpose_pattern(lhs);
    const auto &rhs = classes[find(node.children[1])].pattern;
    return analyze_spmm_pattern(lhs, rhs);
  }

  linear_eclass_id add(enode node) {
    node = canonical(node);
    if (const auto found = memo.find(node); found != memo.end())
      return find(found->second);
    const sparse_pattern pattern = analyze(node);
    const linear_eclass_id id = classes.size();
    classes.push_back({id, 0, pattern, {node}});
    memo.emplace(node, id);
    ++revision;
    return id;
  }

  linear_eclass_id merge(linear_eclass_id lhs, linear_eclass_id rhs) {
    lhs = find(lhs);
    rhs = find(rhs);
    if (lhs == rhs) return lhs;
    if (classes[lhs].pattern != classes[rhs].pattern)
      throw std::invalid_argument("linear e-class sparsity mismatch");
    if (classes[lhs].rank < classes[rhs].rank) std::swap(lhs, rhs);
    classes[rhs].parent = lhs;
    if (classes[lhs].rank == classes[rhs].rank) ++classes[lhs].rank;
    classes[lhs].nodes.insert(classes[lhs].nodes.end(),
                              classes[rhs].nodes.begin(),
                              classes[rhs].nodes.end());
    classes[rhs].nodes.clear();
    ++revision;
    return lhs;
  }

  void rebuild() {
    for (;;) {
      for (size_t id = 0; id < classes.size(); ++id) {
        const size_t root = find(id);
        if (root == id || classes[id].nodes.empty()) continue;
        classes[root].nodes.insert(classes[root].nodes.end(),
                                   classes[id].nodes.begin(),
                                   classes[id].nodes.end());
        classes[id].nodes.clear();
      }
      memo.clear();
      bool congruent = false;
      for (size_t id = 0; id < classes.size() && !congruent; ++id) {
        if (find(id) != id) continue;
        std::vector<enode> unique;
        std::unordered_set<enode, enode_hash> local;
        for (auto node : classes[id].nodes) {
          node = canonical(node);
          if (!local.insert(node).second) continue;
          if (const auto found = memo.find(node); found != memo.end() &&
              find(found->second) != id) {
            merge(id, found->second);
            congruent = true;
            break;
          }
          memo[node] = id;
          unique.push_back(node);
        }
        if (!congruent) classes[id].nodes = std::move(unique);
      }
      if (!congruent) return;
    }
  }

  bool has_identity(linear_eclass_id id) {
    id = find(id);
    return std::ranges::any_of(classes[id].nodes, [](const enode &node) {
      return node.op == linear_egraph_op::identity;
    });
  }

  size_t root_classes() const {
    size_t result = 0;
    for (size_t id = 0; id < classes.size(); ++id)
      result += find_const(id) == id;
    return result;
  }

  size_t node_count() const {
    size_t result = 0;
    for (size_t id = 0; id < classes.size(); ++id)
      if (find_const(id) == id) result += classes[id].nodes.size();
    return result;
  }
};

linear_egraph::linear_egraph() : impl_(std::make_unique<impl>()) {}
linear_egraph::linear_egraph(linear_egraph &&) noexcept = default;
linear_egraph &linear_egraph::operator=(linear_egraph &&) noexcept = default;
linear_egraph::~linear_egraph() = default;

linear_eclass_id linear_egraph::add_leaf(size_t leaf, matrix_layout layout,
                                         sparse_pattern pattern) {
  if (pattern.colind.empty()) pattern = analyze_pattern(layout);
  pattern.validate();
  if (pattern.rows != layout.rows || pattern.cols != layout.cols)
    throw std::invalid_argument("linear e-graph leaf shape mismatch");
  spmm_operand operand{std::move(layout), false, std::move(pattern)};
  if (const auto found = impl_->leaves.find(leaf);
      found != impl_->leaves.end() && found->second != operand)
    throw std::invalid_argument("conflicting linear e-graph leaf");
  impl_->leaves.insert_or_assign(leaf, std::move(operand));
  return impl_->add({linear_egraph_op::leaf, leaf, {}, 0});
}

linear_eclass_id linear_egraph::add_identity(size_t dimension) {
  if (!dimension) throw std::invalid_argument("zero-dimensional identity");
  return impl_->add({linear_egraph_op::identity, dimension, {}, 0});
}

linear_eclass_id linear_egraph::add_multiply(linear_eclass_id lhs,
                                             linear_eclass_id rhs) {
  return impl_->add({linear_egraph_op::multiply, 0, {lhs, rhs}, 2});
}

linear_eclass_id linear_egraph::add_transpose(linear_eclass_id value) {
  return impl_->add({linear_egraph_op::transpose, 0, {value, 0}, 1});
}

linear_eclass_id linear_egraph::merge(linear_eclass_id lhs,
                                      linear_eclass_id rhs) {
  return impl_->merge(lhs, rhs);
}

linear_saturation_result linear_egraph::saturate(size_t max_iterations,
                                                 size_t max_nodes) {
  linear_saturation_result result;
  for (; result.iterations < max_iterations; ++result.iterations) {
    const size_t before = impl_->revision;
    std::vector<std::pair<linear_eclass_id, std::vector<enode>>> snapshot;
    for (size_t id = 0; id < impl_->classes.size(); ++id)
      if (impl_->find(id) == id)
        snapshot.emplace_back(id, impl_->classes[id].nodes);
    for (const auto &[raw_id, nodes] : snapshot) {
      linear_eclass_id id = impl_->find(raw_id);
      for (const auto &raw_node : nodes) {
        const enode node = impl_->canonical(raw_node);
        if (node.op == linear_egraph_op::multiply) {
          const auto left_nodes =
              impl_->classes[impl_->find(node.children[0])].nodes;
          for (const auto &left : left_nodes)
            if (left.op == linear_egraph_op::multiply) {
              const auto tail = add_multiply(left.children[1], node.children[1]);
              merge(id, add_multiply(left.children[0], tail));
            }
          const auto right_nodes =
              impl_->classes[impl_->find(node.children[1])].nodes;
          for (const auto &right : right_nodes)
            if (right.op == linear_egraph_op::multiply) {
              const auto head = add_multiply(node.children[0], right.children[0]);
              merge(id, add_multiply(head, right.children[1]));
            }
          if (impl_->has_identity(node.children[0])) merge(id, node.children[1]);
          if (impl_->has_identity(node.children[1])) merge(id, node.children[0]);
          for (const auto &left : left_nodes) {
            if (left.op != linear_egraph_op::transpose) continue;
            for (const auto &right : right_nodes) {
              if (right.op != linear_egraph_op::transpose) continue;
              const auto product =
                  add_multiply(right.children[0], left.children[0]);
              merge(id, add_transpose(product));
            }
          }
        } else if (node.op == linear_egraph_op::transpose) {
          const auto child_nodes =
              impl_->classes[impl_->find(node.children[0])].nodes;
          for (const auto &child : child_nodes) {
            if (child.op == linear_egraph_op::transpose)
              merge(id, child.children[0]);
            else if (child.op == linear_egraph_op::multiply) {
              const auto lhs = add_transpose(child.children[1]);
              const auto rhs = add_transpose(child.children[0]);
              merge(id, add_multiply(lhs, rhs));
            } else if (child.op == linear_egraph_op::identity)
              merge(id, node.children[0]);
          }
        }
        id = impl_->find(id);
        if (impl_->node_count() >= max_nodes) break;
      }
      if (impl_->node_count() >= max_nodes) break;
    }
    impl_->rebuild();
    if (impl_->node_count() >= max_nodes) break;
    if (impl_->revision == before) {
      result.saturated = true;
      ++result.iterations;
      break;
    }
  }
  result.classes = impl_->root_classes();
  result.nodes = impl_->node_count();
  return result;
}

linear_extraction linear_egraph::extract(linear_eclass_id raw_root) {
  impl_->rebuild();
  const linear_eclass_id root = impl_->find(raw_root);
  std::vector<std::optional<linear_extracted_term>> best(impl_->classes.size());
  bool changed;
  do {
    changed = false;
    for (size_t id = 0; id < impl_->classes.size(); ++id) {
      if (impl_->find(id) != id) continue;
      for (const auto &node : impl_->classes[id].nodes) {
        linear_extracted_term candidate;
        candidate.op = node.op;
        candidate.leaf = node.payload;
        candidate.children = node.children;
        candidate.arity = node.arity;
        bool ready = true;
        for (size_t i = 0; i < node.arity; ++i) {
          candidate.children[i] = impl_->find(node.children[i]);
          ready = ready && best[candidate.children[i]].has_value();
        }
        if (!ready) continue;
        if (node.op == linear_egraph_op::leaf) {
          candidate.value = impl_->leaves.at(node.payload);
        } else if (node.op == linear_egraph_op::identity) {
          candidate.value = {identity_layout(node.payload), false,
                             identity_pattern(node.payload)};
        } else if (node.op == linear_egraph_op::transpose) {
          candidate.value = best[candidate.children[0]]->value;
          candidate.value.transpose = !candidate.value.transpose;
          candidate.cost = best[candidate.children[0]]->cost;
        } else {
          const auto &left = *best[candidate.children[0]];
          const auto &right = *best[candidate.children[1]];
          const auto analysis = analyze_spmm(left.value, right.value);
          candidate.value = {analysis.output_layout, false,
                             analysis.output_pattern};
          candidate.cost = add_cost(left.cost, right.cost);
          candidate.cost.scalar_products += analysis.scalar_products();
          candidate.cost.panel_products += analysis.products.size();
          candidate.cost.temporary_scalars +=
              storage_size(analysis.output_layout);
          ++candidate.cost.operations;
        }
        if (!best[id] || candidate.cost < best[id]->cost) {
          best[id] = std::move(candidate);
          changed = true;
        }
      }
    }
  } while (changed);
  if (!best[root]) throw std::runtime_error("cannot extract linear e-graph");
  return {root, std::move(best)};
}

size_t linear_egraph::classes() const { return impl_->root_classes(); }
size_t linear_egraph::nodes() const { return impl_->node_count(); }

} // namespace moto::linear_backend
