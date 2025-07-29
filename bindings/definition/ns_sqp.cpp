#include <moto/solver/ns_sqp.hpp>
#include <nanobind/stl/function.h>
#include <nanobind/stl/list.h>
#include <type_cast.hpp>

void register_submodule_ns_sqp(nb::module_ &m) {
    using namespace moto;
    using graph_type = decltype(ns_sqp::graph_);
    using binary_func_type = std::function<void(ns_sqp::data *, ns_sqp::data *)>;
    using unary_func_type = std::function<void(ns_sqp::data *)>;

    nb::class_<ns_sqp> sqp(m, "ns_sqp");
    sqp.def(nb::init<>())
        .def_prop_ro("graph", [](ns_sqp &self) -> auto & { return self.graph_; })
        .def("update", [](ns_sqp &self, size_t n_iter) {
            nb::gil_scoped_release rel;
            self.update(n_iter); }, nb::arg("n_iter") = 1, "Update the SQP solver for a given number of iterations")
        .def("forward", &ns_sqp::forward)
        .def("create_node", &ns_sqp::create_node, nb::arg("formulation"), nb::rv_policy::reference, "Create a new node in the SQP graph with the given OCP problem formulation");

    nb::class_<ns_sqp::node_type>(sqp, "node_type")
        .def_prop_ro("addr", [](ns_sqp::node_type &self) { return fmt::format("{:p}", static_cast<const void *>(self.data_)); }, "Get the data address associated with this node");

    nb::class_<ns_sqp::data, node_data>(sqp, "data_type")
        .def_prop_ro("addr", [](ns_sqp::data &self) { return fmt::format("{:p}", static_cast<const void *>(&self)); }, "Get the data address associated with this node")
        .def(nb::init<ocp_ptr_t>(), nb::arg("prob"), "Constructor for ns_sqp data with OCP problem");

    nb::class_<graph_type>(sqp, "graph_type")
        .def(nb::init<>())
        .def("add", &graph_type::add, nb::arg("node"), "Add a node to the graph and return a reference to it", nb::rv_policy::reference)
        .def("set_head", &graph_type::set_head, nb::arg("node"), nb::rv_policy::reference)
        .def("set_tail", &graph_type::set_tail, nb::arg("node"), nb::rv_policy::reference)
        .def("add_edge", &graph_type::add_edge, nb::arg("from"), nb::arg("to"), nb::arg("steps") = 1,
             "Add an edge from one node to another with a given number of steps")
        .def("flatten_nodes", &graph_type::flatten_nodes, nb::rv_policy::reference, "Get the unordered flattened list of all nodes in the graph")
        .def("for_each_parallel", [](graph_type &self, const unary_func_type &callback) {
            nb::gil_scoped_release rel;
            self.for_each_parallel(callback); }, nb::arg("callback"), "Apply a unary function to all nodes in parallel")
        .def("apply_forward_unary", [](graph_type &self, unary_func_type &&callback) {
            nb::gil_scoped_release rel;
            self.apply_forward(std::move(callback)); }, nb::arg("callback"), "Apply a unary function [cur] to all nodes in forward direction")
        .def("apply_forward_binary", [](graph_type &self, binary_func_type &&callback) {
            nb::gil_scoped_release rel;
            self.apply_forward<true, binary_func_type>(std::move(callback)); }, nb::arg("callback"), "Apply a binary function [cur, next] to all nodes in forward direction")
        .def("apply_backward_binary", [](graph_type &self, binary_func_type &&callback) {
            nb::gil_scoped_release rel;
            self.apply_backward<true, binary_func_type>(std::move(callback)); }, nb::arg("callback"), "Apply a binary function [cur, prev] to all nodes in backward direction")
        .def_prop_ro("nodes", &graph_type::nodes, nb::rv_policy::reference);
}