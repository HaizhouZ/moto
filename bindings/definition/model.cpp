#include <moto/model/graph_model.hpp>

#include <type_cast.hpp>

namespace {
auto cast_var_list(const nb::list &items) {
    moto::var_inarg_list out;
    out.reserve(items.size());
    for (auto item : items) {
        out.emplace_back(static_cast<moto::sym &>(*moto::get_expr_ptr(item)));
    }
    return out;
}

nb::list to_py_list(const moto::var_list &items) {
    nb::list out;
    for (const auto &item : items) {
        out.append(nb::cast(item));
    }
    return out;
}
} // namespace

void register_submodule_model(nb::module_ &m) {
    using namespace moto;
    using namespace moto::model;

    nb::class_<model_node, node_ocp>(m, "model_node")
        .def_prop_ro("id", &model_node::id)
        .def_prop_ro("prob", [](const model_node_ptr_t &self) { return self; })
        .def_prop_ro("x", [](const model_node &self) { return to_py_list(self.x()); })
        .def_prop_ro("u", [](const model_node &self) { return to_py_list(self.u()); })
        .def("compose", &model_node::compose)
        .def("compose_terminal", &model_node::compose_terminal);

    nb::class_<model_edge, edge_ocp>(m, "model_edge")
        .def_prop_ro("id", &model_edge::id)
        .def_prop_ro("prob", [](const model_edge_ptr_t &self) { return self; })
        .def_prop_ro("st", &model_edge::st)
        .def_prop_ro("ed", &model_edge::ed)
        .def_prop_ro("x", [](const model_edge &self) { return to_py_list(self.x()); })
        .def_prop_ro("u", [](const model_edge &self) { return to_py_list(self.u()); })
        .def_prop_ro("y", [](const model_edge &self) { return to_py_list(self.y()); })
        .def("compose", &model_edge::compose);

    nb::class_<graph_model>(m, "graph_model")
        .def(nb::init<>())
        .def(
            "add_node",
            [](graph_model &self, const nb::list &x, const nb::list &u) {
                return self.add_node(node_ocp::create(), cast_var_list(x), cast_var_list(u));
            },
            nb::arg("x"),
            nb::arg("u") = nb::list{})
        .def(
            "add_node",
            [](graph_model &self, const node_ocp_ptr_t &prob, const nb::list &x, const nb::list &u) {
                return self.add_node(prob, cast_var_list(x), cast_var_list(u));
            },
            nb::arg("prob") = node_ocp::create(),
            nb::arg("x") = nb::list{},
            nb::arg("u") = nb::list{})
        .def(
            "connect",
            [](graph_model &self, const model_node_ptr_t &st, const model_node_ptr_t &ed, const edge_ocp_ptr_t &prob) {
                return self.connect(st, ed, prob);
            },
            nb::arg("st"),
            nb::arg("ed"),
            nb::arg("prob") = edge_ocp::create())
        .def("compose", [](graph_model &self, const model_edge_ptr_t &edge) { return self.compose(edge); }, nb::arg("edge"))
        .def("compose_terminal", [](graph_model &self, const model_node_ptr_t &node) { return self.compose_terminal(node); }, nb::arg("node"))
        .def_prop_ro("num_nodes", &graph_model::num_nodes)
        .def_prop_ro("num_edges", &graph_model::num_edges);
}
