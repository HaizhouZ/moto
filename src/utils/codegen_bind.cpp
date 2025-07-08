#include <moto/utils/codegen_bind.hpp>
#include <pybind11/embed.h> // everything needed for embedding
#include <pybind11/stl.h>   // For std::vector, std::map, etc. if you pass them
#include <thread>
namespace moto {
namespace utils {

namespace py = pybind11;

std::future<void> generate_n_compile(const std::string &func_name, const std::vector<sym>& in_args, const cs::SX &out,
                                     bool gen_eval, bool gen_jacobian, bool gen_hessian) {
    struct gen_module {
        std::mutex mutex_;
        // use pointer to avoid destruction before main ends (annoying)
        py::scoped_interpreter *g;
        py::module_ *py_moto;
        py::module_ *py_cs;
        gen_module() {
            g = new py::scoped_interpreter();
            py_moto = new py::module_(py::module_::import("moto"));
            py_cs = new py::module_(py::module_::import("casadi"));
        }
    };
    static gen_module g;
    py::object p; // process handle
    {
        py::gil_scoped_acquire acquire;
        std::vector<cs::SX> sx_in_args;
        std::vector<py::object> py_sx_in_args;
        for (auto &a : in_args) {
            // create the object and store its reference immediately
            py::object &new_obj = py_sx_in_args.emplace_back(g.py_cs->attr("SX").attr("sym")(a->name_, a->dim_));
            sx_in_args.emplace_back(a);
            new_obj.attr("name") = a->name_;
            new_obj.attr("field") = g.py_moto->attr("field")((int)a->field_);
        }
        cs::Function tmp_func(func_name + "_tmp", sx_in_args, {out});
        auto py_func = g.py_cs->attr("Function").attr("deserialize")(tmp_func.serialize()); // cs function in python
        py::list in_args_pylist = py::cast(py_sx_in_args);
        auto py_cs_out = py_func.attr("call")(in_args_pylist); // sx output in python
        // process
        p = g.py_moto->attr("generate_and_compile")(func_name, in_args_pylist, py_cs_out.cast<py::list>()[0],
                                                    py::arg("gen_eval") = gen_eval,
                                                    py::arg("gen_jacobian") = gen_jacobian,
                                                    py::arg("gen_hessian") = gen_hessian,
                                                    py::arg("ret_process") = true,
                                                    py::arg("print_level") = 2);
        py::gil_scoped_release release;
    }
    return std::async(std::launch::deferred, [p, func_name]() {
        py::gil_scoped_acquire acquire;
        p();
        py::gil_scoped_release release;
        fmt::print("codegen done for {}.\n", func_name);
    });
}
} // namespace utils
} // namespace moto