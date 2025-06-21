#ifndef __ATRI_OCP_SHOOTING_NODE__
#define __ATRI_OCP_SHOOTING_NODE__
#include <atri/core/directed_graph.hpp>
#include <atri/ocp/data_mgr.hpp>

namespace atri {
// /**
//  * @brief shooting node in an OCP
//  * it will acquire data from data_mgr and release it back on destruction (if not moved)
//  */
template <typename data_type>
    requires std::is_base_of_v<node_data, data_type>
struct shooting_node : public directed_graph_types::node_type<data_type, shooting_node<data_type>> {
    using base = directed_graph_types::node_type<data_type, shooting_node<data_type>>;
    /**
     * @brief Construct a new shooting node object
     *
     * @param formulation problem formulation of this shootin gnode
     * @param mem data management, make sure the data_type is correct
     */
    shooting_node(const problem_ptr_t &formulation)
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

} // namespace atri

#endif /*__NODE_*/