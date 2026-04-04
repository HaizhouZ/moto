#ifndef MOTO_OCP_PROJECTOR_LAYOUT_HPP
#define MOTO_OCP_PROJECTOR_LAYOUT_HPP

#include <string>

#include <moto/ocp/sym.hpp>

namespace moto {
class ocp_base;
class projector_group;

struct projector_hard_constraint_block {
    field_t field = __eq_xu;
    size_t source_begin = 0;
    size_t source_count = 0;
    size_t group_id = 0;
    std::string group;
};

class projector_layout {
    ocp_base *owner_ = nullptr;

  public:
    projector_layout() = default;
    explicit projector_layout(ocp_base *owner)
        : owner_(owner) {}

    projector_group group();
};

class projector_group {
    ocp_base *owner_ = nullptr;
    size_t id_ = 0;

  public:
    projector_group() = default;
    projector_group(ocp_base *owner, size_t id)
        : owner_(owner), id_(id) {}

    projector_group &require_primal(const var_inarg_list &vars);
    projector_group &require_primal(const var_list &vars) {
        return require_primal(var_inarg_list(vars));
    }
    projector_group &require_constraint(const expr_inarg_list &exprs);
    projector_group &require_constraint(const expr_list &exprs) {
        return require_constraint(expr_inarg_list(exprs));
    }
    projector_group &require_before(const projector_group &after);
    projector_group &require_after(const projector_group &before);
    size_t id() const { return id_; }
};
} // namespace moto

#endif // MOTO_OCP_PROJECTOR_LAYOUT_HPP
