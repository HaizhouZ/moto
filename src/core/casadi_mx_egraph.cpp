#include "casadi_mx_egraph.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

namespace moto::linear_backend::detail {
namespace {

using class_id = size_t;
enum class op { leaf, identity, multiply, transpose };

struct cost {
  size_t scalar_products = 0, panel_products = 0;
  size_t temporary_scalars = 0, operations = 0;
  bool operator<(const cost &other) const {
    return std::tie(scalar_products, panel_products, temporary_scalars,
                    operations) <
           std::tie(other.scalar_products, other.panel_products,
                    other.temporary_scalars, other.operations);
  }
  cost &operator+=(const cost &other) {
    scalar_products += other.scalar_products;
    panel_products += other.panel_products;
    temporary_scalars += other.temporary_scalars;
    operations += other.operations;
    return *this;
  }
};

struct node {
  op operation = op::leaf;
  size_t payload = 0, arity = 0;
  std::array<class_id, 2> children{};
  bool operator==(const node &) const = default;
};

struct node_hash {
  size_t operator()(const node &value) const noexcept {
    size_t seed = static_cast<size_t>(value.operation) * size_t{0x9e3779b9} +
                  value.payload;
    for (size_t i = 0; i < value.arity; ++i)
      seed ^=
          value.children[i] + size_t{0x9e3779b9} + (seed << 6) + (seed >> 2);
    return seed;
  }
};

struct term {
  op operation = op::leaf;
  size_t leaf = 0, arity = 0;
  std::array<class_id, 2> children{};
  spmm_operand value;
  cost work;
};

struct extraction {
  class_id root = 0;
  std::vector<std::optional<term>> terms;
  const term &at(class_id id) const {
    if (id >= terms.size() || !terms[id])
      throw std::out_of_range("missing extracted linear e-class");
    return *terms[id];
  }
};

size_t storage_size(const matrix_layout &layout) {
  return std::accumulate(layout.panels.begin(), layout.panels.end(), size_t{0},
                         [](size_t total, const panel_layout &panel) {
                           if (panel.pattern == sparsity::eye)
                             return total;
                           return total + (panel.pattern == sparsity::dense
                                               ? panel.rows * panel.cols
                                               : panel.rows);
                         });
}

sparse_pattern identity_pattern(size_t dimension) {
  sparse_pattern result{dimension, dimension, {0}, {}};
  for (size_t i = 0; i < dimension; ++i) {
    result.row.push_back(i);
    result.colind.push_back(i + 1);
  }
  return result;
}

spmm_operand identity_operand(size_t dimension) {
  matrix_layout layout{
      dimension, dimension, {{sparsity::eye, 0, 0, dimension, dimension}}};
  return {std::move(layout), false, identity_pattern(dimension)};
}

class egraph {
  struct eclass {
    class_id parent = 0;
    size_t rank = 0;
    sparse_pattern pattern;
    std::vector<node> nodes;
  };

  std::vector<eclass> classes_;
  std::unordered_map<node, class_id, node_hash> memo_;
  std::unordered_map<size_t, spmm_operand> leaves_;
  size_t revision_ = 0;

  class_id find(class_id id) {
    if (id >= classes_.size())
      throw std::out_of_range("invalid e-class");
    if (classes_[id].parent != id)
      classes_[id].parent = find(classes_[id].parent);
    return classes_[id].parent;
  }
  class_id find(class_id id) const {
    if (id >= classes_.size())
      throw std::out_of_range("invalid e-class");
    while (classes_[id].parent != id)
      id = classes_[id].parent;
    return id;
  }
  node canonical(node value) {
    for (size_t i = 0; i < value.arity; ++i)
      value.children[i] = find(value.children[i]);
    return value;
  }
  sparse_pattern analyze(const node &value) {
    if (value.operation == op::leaf)
      return leaves_.at(value.payload).pattern;
    if (value.operation == op::identity)
      return identity_pattern(value.payload);
    const auto &lhs = classes_[find(value.children[0])].pattern;
    if (value.operation == op::transpose)
      return transpose_pattern(lhs);
    return analyze_spmm_pattern(lhs, classes_[find(value.children[1])].pattern);
  }
  class_id add(node value) {
    value = canonical(value);
    if (const auto found = memo_.find(value); found != memo_.end())
      return find(found->second);
    const class_id id = classes_.size();
    classes_.push_back({id, 0, analyze(value), {value}});
    memo_.emplace(value, id);
    ++revision_;
    return id;
  }
  bool has_identity(class_id id) {
    const auto &nodes = classes_[find(id)].nodes;
    return std::ranges::any_of(nodes, [](const node &value) {
      return value.operation == op::identity;
    });
  }
  size_t node_count() const {
    size_t result = 0;
    for (class_id id = 0; id < classes_.size(); ++id)
      if (find(id) == id)
        result += classes_[id].nodes.size();
    return result;
  }
  void rebuild() {
    for (;;) {
      for (class_id id = 0; id < classes_.size(); ++id) {
        const class_id root = find(id);
        if (root == id || classes_[id].nodes.empty())
          continue;
        classes_[root].nodes.insert(classes_[root].nodes.end(),
                                    classes_[id].nodes.begin(),
                                    classes_[id].nodes.end());
        classes_[id].nodes.clear();
      }
      memo_.clear();
      bool congruent = false;
      for (class_id id = 0; id < classes_.size() && !congruent; ++id) {
        if (find(id) != id)
          continue;
        std::vector<node> unique;
        std::unordered_set<node, node_hash> local;
        for (auto value : classes_[id].nodes) {
          value = canonical(value);
          if (!local.insert(value).second)
            continue;
          if (const auto found = memo_.find(value);
              found != memo_.end() && find(found->second) != id) {
            merge(id, found->second);
            congruent = true;
            break;
          }
          memo_[value] = id;
          unique.push_back(value);
        }
        if (!congruent)
          classes_[id].nodes = std::move(unique);
      }
      if (!congruent)
        return;
    }
  }

public:
  class_id leaf(size_t key, matrix_layout layout, sparse_pattern pattern) {
    if (pattern.colind.empty())
      pattern = analyze_pattern(layout);
    pattern.validate();
    if (pattern.rows != layout.rows || pattern.cols != layout.cols)
      throw std::invalid_argument("linear e-graph leaf shape mismatch");
    spmm_operand value{std::move(layout), false, std::move(pattern)};
    if (const auto found = leaves_.find(key);
        found != leaves_.end() && found->second != value)
      throw std::invalid_argument("conflicting linear e-graph leaf");
    leaves_.insert_or_assign(key, std::move(value));
    return add({op::leaf, key, 0, {}});
  }
  class_id identity(size_t dimension) {
    if (!dimension)
      throw std::invalid_argument("zero-dimensional identity");
    return add({op::identity, dimension, 0, {}});
  }
  class_id multiply(class_id lhs, class_id rhs) {
    return add({op::multiply, 0, 2, {lhs, rhs}});
  }
  class_id transpose(class_id value) {
    return add({op::transpose, 0, 1, {value, 0}});
  }
  class_id merge(class_id lhs, class_id rhs) {
    lhs = find(lhs);
    rhs = find(rhs);
    if (lhs == rhs)
      return lhs;
    if (classes_[lhs].pattern != classes_[rhs].pattern)
      throw std::invalid_argument("linear e-class sparsity mismatch");
    if (classes_[lhs].rank < classes_[rhs].rank)
      std::swap(lhs, rhs);
    classes_[rhs].parent = lhs;
    if (classes_[lhs].rank == classes_[rhs].rank)
      ++classes_[lhs].rank;
    classes_[lhs].nodes.insert(classes_[lhs].nodes.end(),
                               classes_[rhs].nodes.begin(),
                               classes_[rhs].nodes.end());
    classes_[rhs].nodes.clear();
    ++revision_;
    return lhs;
  }
  size_t saturate(size_t max_iterations = 16, size_t max_nodes = 50000) {
    for (size_t iteration = 0; iteration < max_iterations; ++iteration) {
      const size_t before = revision_;
      std::vector<std::pair<class_id, std::vector<node>>> snapshot;
      for (class_id id = 0; id < classes_.size(); ++id)
        if (find(id) == id)
          snapshot.emplace_back(id, classes_[id].nodes);
      for (const auto &[raw_id, nodes] : snapshot) {
        class_id id = find(raw_id);
        for (const auto &raw : nodes) {
          const node value = canonical(raw);
          if (value.operation == op::multiply) {
            const auto left = classes_[find(value.children[0])].nodes;
            const auto right = classes_[find(value.children[1])].nodes;
            for (const auto &item : left)
              if (item.operation == op::multiply)
                merge(id,
                      multiply(item.children[0],
                               multiply(item.children[1], value.children[1])));
            for (const auto &item : right)
              if (item.operation == op::multiply)
                merge(id,
                      multiply(multiply(value.children[0], item.children[0]),
                               item.children[1]));
            if (has_identity(value.children[0]))
              merge(id, value.children[1]);
            if (has_identity(value.children[1]))
              merge(id, value.children[0]);
            for (const auto &lhs : left) {
              if (lhs.operation != op::transpose)
                continue;
              for (const auto &rhs : right)
                if (rhs.operation == op::transpose)
                  merge(id,
                        transpose(multiply(rhs.children[0], lhs.children[0])));
            }
          } else if (value.operation == op::transpose) {
            const auto children = classes_[find(value.children[0])].nodes;
            for (const auto &child : children) {
              if (child.operation == op::transpose)
                merge(id, child.children[0]);
              else if (child.operation == op::multiply)
                merge(id, multiply(transpose(child.children[1]),
                                   transpose(child.children[0])));
              else if (child.operation == op::identity)
                merge(id, value.children[0]);
            }
          }
          id = find(id);
          if (node_count() >= max_nodes)
            break;
        }
        if (node_count() >= max_nodes)
          break;
      }
      rebuild();
      if (node_count() >= max_nodes || revision_ == before)
        break;
    }
    return node_count();
  }
  extraction extract(class_id raw_root) {
    rebuild();
    const class_id root = find(raw_root);
    std::vector<std::optional<term>> best(classes_.size());
    bool changed;
    do {
      changed = false;
      for (class_id id = 0; id < classes_.size(); ++id) {
        if (find(id) != id)
          continue;
        for (const auto &value : classes_[id].nodes) {
          term candidate{value.operation, value.payload, value.arity,
                         value.children};
          bool ready = true;
          for (size_t i = 0; i < value.arity; ++i) {
            candidate.children[i] = find(value.children[i]);
            ready = ready && best[candidate.children[i]].has_value();
          }
          if (!ready)
            continue;
          if (value.operation == op::leaf) {
            candidate.value = leaves_.at(value.payload);
          } else if (value.operation == op::identity) {
            candidate.value = identity_operand(value.payload);
          } else if (value.operation == op::transpose) {
            candidate.value = best[candidate.children[0]]->value;
            candidate.value.transpose = !candidate.value.transpose;
            candidate.work = best[candidate.children[0]]->work;
          } else {
            const auto &lhs = *best[candidate.children[0]];
            const auto &rhs = *best[candidate.children[1]];
            const auto product = analyze_spmm(lhs.value, rhs.value);
            candidate.value = {product.output_layout, false,
                               product.output_pattern};
            candidate.work = lhs.work;
            candidate.work += rhs.work;
            candidate.work.scalar_products += product.scalar_products();
            candidate.work.panel_products += product.products.size();
            candidate.work.temporary_scalars +=
                storage_size(product.output_layout);
            ++candidate.work.operations;
          }
          if (!best[id] || candidate.work < best[id]->work) {
            best[id] = std::move(candidate);
            changed = true;
          }
        }
      }
    } while (changed);
    if (!best[root])
      throw std::runtime_error("cannot extract linear e-graph");
    return {root, std::move(best)};
  }
};

bool is_product(const casadi::MX &value) {
  return value.op() == casadi::OP_MTIMES && value.n_dep() == 3 &&
         value.dep(0).is_zero();
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
  const std::unordered_map<const casadi::MXNode *, matrix_layout> &layouts;
  egraph graph;
  std::unordered_map<const casadi::MXNode *, class_id> imported;
  std::unordered_map<const casadi::MXNode *, size_t> leaf_ids;
  std::vector<casadi::MX> leaves;

  class_id add(const casadi::MX &value) {
    if (const auto found = imported.find(value.get()); found != imported.end())
      return found->second;
    class_id id;
    if (is_product(value))
      id = graph.multiply(add(value.dep(1)), add(value.dep(2)));
    else if (value.op() == casadi::OP_TRANSPOSE)
      id = graph.transpose(add(value.dep(0)));
    else if (value.is_eye())
      id = graph.identity(value.size1());
    else {
      const auto [found, inserted] =
          leaf_ids.emplace(value.get(), leaves.size());
      if (inserted)
        leaves.push_back(value);
      const sparse_pattern exact = pattern(value.sparsity());
      const auto layout = layouts.find(value.get());
      id = graph.leaf(found->second,
                      layout == layouts.end() ? panelize_pattern(exact)
                                              : layout->second,
                      exact);
    }
    imported.emplace(value.get(), id);
    return id;
  }

  casadi::MX optimize(const casadi::MX &root) {
    const class_id root_id = add(root);
    const cost original = graph.extract(root_id).at(root_id).work;
    const size_t nodes = graph.saturate();
    const extraction result = graph.extract(root_id);
    if (std::getenv("MOTO_TRACE_EGRAPH") &&
        result.at(result.root).work < original) {
      const auto &optimized = result.at(result.root).work;
      std::cerr << "linear e-graph " << root.size1() << 'x' << root.size2()
                << " nodes=" << nodes
                << " scalar_products=" << original.scalar_products << "->"
                << optimized.scalar_products
                << " panel_products=" << original.panel_products << "->"
                << optimized.panel_products
                << " temporaries=" << original.temporary_scalars << "->"
                << optimized.temporary_scalars << '\n';
    }
    std::unordered_map<class_id, casadi::MX> values;
    const auto build = [&](const auto &self, class_id id) -> casadi::MX {
      if (const auto found = values.find(id); found != values.end())
        return found->second;
      const term &value = result.at(id);
      casadi::MX expression;
      if (value.operation == op::leaf)
        expression = leaves.at(value.leaf);
      else if (value.operation == op::identity)
        expression = casadi::MX::eye(value.value.rows());
      else if (value.operation == op::transpose)
        expression = self(self, value.children[0]).T();
      else
        expression = casadi::MX::mtimes(self(self, value.children[0]),
                                        self(self, value.children[1]));
      values.emplace(id, expression);
      return expression;
    };
    return build(build, result.root);
  }
};

using parent_map =
    std::unordered_map<const casadi::MXNode *, std::vector<casadi::MX>>;

void collect(const casadi::MX &value,
             std::unordered_set<const casadi::MXNode *> &visited,
             parent_map &parents, std::vector<casadi::MX> &algebraic) {
  if (!value.get() || !visited.insert(value.get()).second)
    return;
  const bool supported =
      is_product(value) || value.op() == casadi::OP_TRANSPOSE;
  if (supported)
    algebraic.push_back(value);
  for (casadi_int i = 0; i < value.n_dep(); ++i) {
    const casadi::MX dependency = value.dep(i);
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
    layouts.emplace(inputs[i].get(),
                    supplied ? input_layouts[i] : panelize_pattern(exact));
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
          return !is_product(parent) && parent.op() != casadi::OP_TRANSPOSE;
        }))
      targets.push_back(value);
  }
  if (targets.empty())
    return output_entries;

  std::vector<casadi::MX> replacements;
  replacements.reserve(targets.size());
  for (const auto &target : targets)
    replacements.push_back(importer{layouts}.optimize(target));
  auto result = output_entries;
  for (auto &entry : result)
    entry = casadi::MX::graph_substitute(entry, targets, replacements);
  return result;
}

} // namespace moto::linear_backend::detail
