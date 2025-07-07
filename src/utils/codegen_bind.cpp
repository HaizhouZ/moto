#include <moto/utils/codegen_bind.hpp>
#include <pybind11/embed.h> // everything needed for embedding
#include <pybind11/stl.h>   // For std::vector, std::map, etc. if you pass them
#include <thread>
namespace moto {
namespace utils {

namespace py = pybind11;

void generate_n_compile(const std::string &func_name, std::vector<sym> in_args, const cs::SX &out,
                        bool gen_eval, bool gen_jacobian, bool gen_hessian) {
    struct gen_module {
        std::mutex mutex_;
        py::module_ *py_moto;
        py::module_ *py_cs;
        gen_module() {
            py::initialize_interpreter();
            py_moto = new py::module_(py::module_::import("moto"));
            py_cs = new py::module_(py::module_::import("casadi"));
        }
        ~gen_module() {
        }
    };
    static gen_module g;
    g.mutex_.lock();
    std::vector<cs::SX> sx_in_args;
    std::vector<py::object> py_sx_in_args;
    py::gil_scoped_acquire acquire;
    for (auto &a : in_args) {
        // create the object and store its reference immediately
        py::object &new_obj = py_sx_in_args.emplace_back(g.py_cs->attr("SX").attr("sym")(a->name_, a->dim_));
        sx_in_args.emplace_back(a);
        new_obj.attr("name") = a->name_;
        new_obj.attr("field") = g.py_moto->attr("field")((int)a->field_);
    }
    cs::Function tmp_func("tmp_func", sx_in_args, {out});
    auto py_func = g.py_cs->attr("Function").attr("deserialize")(tmp_func.serialize()); // cs function in python
    py::list in_args_pylist = py::cast(py_sx_in_args);
    auto py_cs_out = py_func.attr("call")(in_args_pylist); // sx output in python
    // process
    auto p = g.py_moto->attr("generate_and_compile")(func_name, in_args_pylist, py_cs_out.cast<py::list>()[0],
                                                     py::arg("gen_eval") = gen_eval,
                                                     py::arg("gen_jacobian") = gen_jacobian,
                                                     py::arg("gen_hessian") = gen_hessian,
                                                     py::arg("ret_process") = true,
                                                     py::arg("print_level") = 1);
    g.mutex_.unlock();
    for (auto &p_ : p.cast<py::list>())
        p_.attr("join")();
    py::gil_scoped_release release;
}
} // namespace utils
} // namespace moto