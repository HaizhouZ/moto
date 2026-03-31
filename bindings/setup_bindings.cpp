#include <binding_fwd.hpp>

extern void register_submodule_codegen(nb::module_ &m);
extern void register_submodule_fields(nb::module_ &m);
extern void register_submodule_node_data(nb::module_ &m);
extern void register_submodule_functional(nb::module_ &m);
extern void register_submodule_model(nb::module_ &m);
extern void register_submodule_ns_sqp(nb::module_ &m);
extern void register_submodule_multibody(nb::module_ &m);
NB_MODULE(moto_pywrap, m) {
    nb::set_leak_warnings(false);
    register_submodule_codegen(m);
    register_submodule_fields(m);
    register_submodule_node_data(m);
    register_submodule_functional(m);
    register_submodule_model(m);
    register_submodule_ns_sqp(m);
    register_submodule_multibody(m);
}
