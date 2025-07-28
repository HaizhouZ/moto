#include <moto/solver/ns_sqp.hpp>
#include <type_cast.hpp>

void register_submodule_ns_sqp(nb::module_ &m) {
    using namespace moto;
    using graph_type = decltype(ns_sqp::graph_);

    nb::class_<ns_sqp> sqp(m, "ns_sqp");
    sqp.def(nb::init<>())
        .def_prop_ro("graph", [](ns_sqp &self) -> auto & { return self.graph_; })
        .def("update", &ns_sqp::update, nb::arg("n_iter") = 1)
        .def("forward", &ns_sqp::forward)
        .def("create_node", &ns_sqp::create_node, nb::arg("formulation"), nb::rv_policy::reference,
             "Create a new node in the SQP graph with the given OCP problem formulation");

    nb::class_<ns_sqp::node_type>(sqp, "node_type")
        .def("data", [](ns_sqp::node_type &self) { return fmt::format("{:p}", static_cast<const void *>(self.data_)); }, "Get the data associated with this node");

    nb::class_<ns_sqp::data, node_data>(sqp, "data_type")
        .def(nb::init<ocp_ptr_t>(), nb::arg("prob"), "Constructor for ns_sqp data with OCP problem");

    nb::class_<graph_type>(sqp, "graph_type")
        .def(nb::init<>())
        .def("add", &graph_type::add, nb::arg("node"), "Add a node to the graph and return a reference to it", nb::rv_policy::reference)
        .def("set_head", &graph_type::set_head, nb::arg("node"), nb::rv_policy::reference)
        .def("set_tail", &graph_type::set_tail, nb::arg("node"), nb::rv_policy::reference)
        .def("add_edge", &graph_type::add_edge, nb::arg("from"), nb::arg("to"), nb::arg("steps") = 1,
             "Add an edge from one node to another with a given number of steps")
        .def_prop_ro("nodes", &graph_type::nodes, nb::rv_policy::reference);
}