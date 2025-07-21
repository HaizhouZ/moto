#include <moto/utils/codegen.hpp>
#include <binding_fwd.hpp>

using namespace moto;

/// @brief A helper struct to extract the PySwigObject from a Python object
/// This is used to access the underlying C++ object pointer from a Python object
/// that has been wrapped with SWIG.
/// The PySwigObject struct is expected to have a 'this' attribute that points to
/// the C++ object pointer.
/// The 'desc' attribute is used to store the type description of the object.
/// @cite https://docs.ros.org/en/noetic/api/eigenpy/html/swig_8hpp_source.html
struct PySwigObject {
    PyObject_HEAD void *ptr;
    const char *desc;
};

inline PySwigObject *get_PySwigObject(PyObject *pyObj) {
    if (!PyObject_HasAttrString(pyObj, "this"))
        return NULL;

    PyObject *this_ptr = PyObject_GetAttrString(pyObj, "this");
    if (this_ptr == NULL)
        return nullptr;
    PySwigObject *swig_obj = reinterpret_cast<PySwigObject *>(this_ptr);

    return swig_obj;
}
namespace nanobind {
namespace detail {
template <>
struct type_caster<cs::SX> {
    NB_TYPE_CASTER(cs::SX, /* type_name_for_error_messages */ const_name("casadi.SX"));

    bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) {
        // Logic to convert Python object (src) to MyCustomType
        // e.g., extract an int from src and set self->value
        value = cs::SX();
        auto swig_obj = get_PySwigObject(src.ptr());
        if (swig_obj) {
            assert(swig_obj != nullptr);
            auto sx_ptr = reinterpret_cast<cs::SX *>(swig_obj->ptr);
            value = *sx_ptr;
        } else
            nb::raise_type_error("Expected a casadi.SX object, but got: %s", nb::type_name(src).c_str());
        return true;
    }

    static handle from_cpp(const cs::SX &src, rv_policy policy, cleanup_list *cleanup) {
        // Logic to convert MyCustomType (src) to a Python object
        // e.g., return a Python int from src.value
        object py_cs_module = nb::module_::import_("casadi");
        object py_cs_sx = py_cs_module.attr("SX")();
        auto swig_obj = get_PySwigObject(py_cs_sx.ptr());
        assert(swig_obj != nullptr);
        auto sx_ptr = reinterpret_cast<cs::SX *>(swig_obj->ptr);
        *sx_ptr = src;
        return py_cs_sx.release();
    }
};
} // namespace detail
} // namespace nanobind

void register_submodule_codegen(nb::module_ &m) {
    nb::class_<utils::cs_codegen::worker>(m, "codegen_worker")
        .def("wait_until_finished", &utils::cs_codegen::worker::wait_until_finished,
             "Wait until the code generation task is finished.");
    nb::class_<utils::cs_codegen::worker_list>(m, "codegen_worker_list")
        .def("wait_until_finished", &utils::cs_codegen::worker_list::wait_until_finished,
             "Wait until all code generation tasks in the list are finished.");
    nb::class_<utils::cs_codegen::task>(m, "codegen_task")
        .def(nb::init<>())
        .def_rw("func_name", &utils::cs_codegen::task::func_name)
        .def_rw("sx_inputs", &utils::cs_codegen::task::sx_inputs)
        .def_rw("sx_output", &utils::cs_codegen::task::sx_output)
        .def_rw("gen_eval", &utils::cs_codegen::task::gen_eval)
        .def_rw("gen_jacobian", &utils::cs_codegen::task::gen_jacobian)
        .def_rw("gen_hessian", &utils::cs_codegen::task::gen_hessian)
        .def_rw("ext_jac", &utils::cs_codegen::task::ext_jac)
        .def_rw("ext_hess", &utils::cs_codegen::task::ext_hess)
        .def_rw("output_dir", &utils::cs_codegen::task::output_dir)
        .def_rw("force_recompile", &utils::cs_codegen::task::force_recompile)
        .def_rw("check_jac_ad", &utils::cs_codegen::task::check_jac_ad)
        .def_rw("append_value", &utils::cs_codegen::task::append_value)
        .def_rw("append_jac", &utils::cs_codegen::task::append_jac)
        .def_rw("keep_generated_src", &utils::cs_codegen::task::keep_generated_src)
        .def_rw("eval_compile_flag", &utils::cs_codegen::task::eval_compile_flag)
        .def_rw("jac_compile_flag", &utils::cs_codegen::task::jac_compile_flag)
        .def_rw("hess_compile_flag", &utils::cs_codegen::task::hess_compile_flag)
        .def_rw("prefix", &utils::cs_codegen::task::prefix)
        .def_rw("verbose", &utils::cs_codegen::task::verbose);

    m.def("generate_and_compile", &utils::cs_codegen::generate_and_compile,
                       nb::arg("task"), "Generate and compile code for the given task.");
    m.def("wait_until_generated", &utils::cs_codegen::wait_until_generated,
                       "Wait until all code generation tasks are finished.");
}
