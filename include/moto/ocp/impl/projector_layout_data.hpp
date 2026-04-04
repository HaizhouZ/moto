#ifndef MOTO_OCP_IMPL_PROJECTOR_LAYOUT_DATA_HPP
#define MOTO_OCP_IMPL_PROJECTOR_LAYOUT_DATA_HPP

#include <string>
#include <vector>

#include <moto/ocp/projector_layout.hpp>

namespace moto {
struct projector_member_ref {
    size_t uid = 0;
    std::string name;
};

struct projector_group_spec {
    size_t id = 0;
    std::vector<projector_member_ref> primals;
    std::vector<projector_member_ref> constraints;
};

struct projector_before_spec {
    size_t before = 0;
    size_t after = 0;
};

struct projector_layout_spec {
    std::vector<projector_group_spec> groups;
    std::vector<projector_before_spec> order;

    bool empty() const { return groups.empty() && order.empty(); }
};

struct compiled_projector_layout_data {
    std::vector<projector_hard_constraint_block> hard_constraint_blocks;

    void clear() { hard_constraint_blocks.clear(); }
    bool empty() const { return hard_constraint_blocks.empty(); }
};
} // namespace moto

#endif // MOTO_OCP_IMPL_PROJECTOR_LAYOUT_DATA_HPP
