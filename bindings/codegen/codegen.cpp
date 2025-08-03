#include <moto/utils/codegen.hpp>
#include <type_cast.hpp>
#include <nanobind/stl/vector.h>

using namespace moto;

void register_submodule_codegen(nb::module_ &m) {
    // nb::class_<utils::cs_codegen::worker>(m, "codegen_worker")
    //     .def("wait_until_finished", &utils::cs_codegen::worker::wait_until_finished,
    //          "Wait until the code generation task is finished.");
    nb::class_<utils::cs_codegen::job_list>(m, "codegen_job_list")
        .def("wait_until_finished", &utils::cs_codegen::job_list::wait_until_finished,
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
