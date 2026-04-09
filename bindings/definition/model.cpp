#include <moto/ocp/graph_model.hpp>
#include <type_cast.hpp>

void register_submodule_model(nb::module_ &m) {
    using namespace moto;
    using namespace moto;

    nb::class_<graph_model>(m, "graph_model")
        .def(nb::init<>())
        .def("reserve",
             &graph_model::reserve,
             nb::arg("node_capacity"),
             nb::arg("edge_capacity"))
        .def("create_node",
             &graph_model::create_node,
             nb::arg("prob") = node_ocp::create())
        .def("create_edge",
             &graph_model::create_edge,
             nb::arg("prob") = edge_ocp::create())
        .def("connect",
             &graph_model::connect,
             nb::arg("st"),
             nb::arg("ed"),
             nb::arg("prob") = edge_ocp::create())
        .def("add_path",
             static_cast<std::vector<model_edge_ptr_t> (graph_model::*)(const model_node_ptr_t &, const model_node_ptr_t &, size_t, const edge_ocp_ptr_t &)>(&graph_model::add_path),
             nb::arg("st"),
             nb::arg("ed"),
             nb::arg("n_edges"),
             nb::arg("prob") = edge_ocp::create())
        .def("add_path",
             [](graph_model &self, const node_ocp_ptr_t &st_prob, const node_ocp_ptr_t &ed_prob, size_t n_edges, const edge_ocp_ptr_t &prob) {
                 return self.add_path(st_prob, ed_prob, n_edges, prob);
             },
             nb::arg("st_prob"),
             nb::arg("ed_prob"),
             nb::arg("n_edges"),
             nb::arg("prob") = edge_ocp::create())
        .def("add_path",
             static_cast<std::vector<model_edge_ptr_t> (graph_model::*)(const model_node_ptr_t &, size_t, const edge_ocp_ptr_t &)>(&graph_model::add_path),
             nb::arg("st"),
             nb::arg("n_edges"),
             nb::arg("prob") = edge_ocp::create())
        .def("add_path",
             [](graph_model &self, const node_ocp_ptr_t &st_prob, size_t n_edges, const edge_ocp_ptr_t &prob) {
                 return self.add_path(st_prob, n_edges, prob);
             },
             nb::arg("st_prob"),
             nb::arg("n_edges"),
             nb::arg("prob") = edge_ocp::create())
        .def("add_path",
             static_cast<std::vector<model_edge_ptr_t> (graph_model::*)(size_t, const edge_ocp_ptr_t &)>(&graph_model::add_path),
             nb::arg("n_edges"),
             nb::arg("prob") = edge_ocp::create())
        .def("compose", &graph_model::compose, nb::arg("edge"))
        .def("compose_all", &graph_model::compose_all)
        .def("compose_terminal", &graph_model::compose_terminal, nb::arg("node"))
        .def_prop_ro("num_nodes", &graph_model::num_nodes)
        .def_prop_ro("num_edges", &graph_model::num_edges);

     m.attr("model_node") = m.attr("node_ocp");
     m.attr("model_edge") = m.attr("edge_ocp");
}
