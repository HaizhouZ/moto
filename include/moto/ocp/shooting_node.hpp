#ifndef __MOTO_OCP_SHOOTING_NODE__
#define __MOTO_OCP_SHOOTING_NODE__
#include <moto/core/directed_graph.hpp>
#include <moto/ocp/data_mgr.hpp>

namespace moto {
/**
 * @brief shooting node in an OCP
 * it will acquire data from data_mgr and release it upon destruction (if not moved)
 */
template <typename T>
    requires std::is_base_of_v<node_data, T>
struct shooting_node : public graph_types::node_base<T, shooting_node<T>> {
    using base = graph_types::node_base<T, shooting_node<T>>;
    using data_type = base::data_type;

    /**
     * @brief Construct a new shooting node object
     *
     * @param formulation problem formulation of this shootin gnode
     */
    shooting_node(const ocp_ptr_t &formulation)
        : mem_(data_mgr::get<data_type>()) {
        base::data_ = dynamic_cast<data_type *>(mem_.acquire(formulation));
    }
    /**
     * @brief Construct a new shooting node sharing the same problem and data_mgr as rhs
     * @note it will not copy the data of rhs but get a new one
     * @param rhs the shooting ndoe to be copied
     */
    shooting_node(const shooting_node &rhs) : base(rhs), mem_(rhs.mem_) {
        base::data_ = dynamic_cast<data_type *>(mem_.acquire(rhs.data_));
    }
    shooting_node(shooting_node &&rhs) : base(std::move(rhs)), mem_(rhs.mem_) {
        base::data_ = rhs.data_;
        rhs.data_ = nullptr; // avoid double release
    }

    ~shooting_node() {
        if (base::data_)
            mem_.release(base::data_);
    }
    data_mgr &mem_;
};

} // namespace moto

#endif /*__NODE_*/