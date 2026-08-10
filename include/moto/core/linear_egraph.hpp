#ifndef MOTO_CORE_LINEAR_EGRAPH_HPP
#define MOTO_CORE_LINEAR_EGRAPH_HPP

#include <moto/core/linear_backend.hpp>

#include <array>
#include <memory>
#include <optional>

namespace moto::linear_backend {

using linear_eclass_id = size_t;

enum class linear_egraph_op { leaf, identity, multiply, transpose };

struct linear_egraph_cost {
  size_t scalar_products = 0;
  size_t panel_products = 0;
  size_t temporary_scalars = 0;
  size_t operations = 0;

  bool operator==(const linear_egraph_cost &) const = default;
  bool operator<(const linear_egraph_cost &other) const;
};

struct linear_extracted_term {
  linear_egraph_op op = linear_egraph_op::leaf;
  size_t leaf = 0;
  std::array<linear_eclass_id, 2> children{};
  size_t arity = 0;
  spmm_operand value;
  linear_egraph_cost cost;
};

struct linear_extraction {
  linear_eclass_id root = 0;
  std::vector<std::optional<linear_extracted_term>> terms;

  const linear_extracted_term &at(linear_eclass_id id) const;
};

struct linear_saturation_result {
  size_t iterations = 0;
  size_t classes = 0;
  size_t nodes = 0;
  bool saturated = false;
};

/// Small typed equality-saturation engine for block linear algebra. The graph
/// owns algebraic alternatives only; concrete storage is selected by extract.
class linear_egraph {
public:
  linear_egraph();
  linear_egraph(linear_egraph &&) noexcept;
  linear_egraph &operator=(linear_egraph &&) noexcept;
  linear_egraph(const linear_egraph &) = delete;
  linear_egraph &operator=(const linear_egraph &) = delete;
  ~linear_egraph();

  linear_eclass_id add_leaf(size_t leaf, matrix_layout layout,
                            sparse_pattern pattern = {});
  linear_eclass_id add_identity(size_t dimension);
  linear_eclass_id add_multiply(linear_eclass_id lhs,
                                linear_eclass_id rhs);
  linear_eclass_id add_transpose(linear_eclass_id value);
  linear_eclass_id merge(linear_eclass_id lhs, linear_eclass_id rhs);

  linear_saturation_result saturate(size_t max_iterations = 16,
                                    size_t max_nodes = 50000);
  linear_extraction extract(linear_eclass_id root);

  size_t classes() const;
  size_t nodes() const;

private:
  struct impl;
  std::unique_ptr<impl> impl_;
};

} // namespace moto::linear_backend

#endif
