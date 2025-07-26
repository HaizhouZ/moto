#include <binding_fwd.hpp>
#include <moto/ocp/constr.hpp>
#include <moto/ocp/cost.hpp>
#include <moto/ocp/pre_comp.hpp>
#include <moto/ocp/sym.hpp>
#include <moto/ocp/usr_func.hpp>

namespace nanobind {
namespace detail {
template <>
struct type_caster<moto::expr_inarg_list> {
    NB_TYPE_CASTER(moto::expr_inarg_list, io_name(NB_TYPING_SEQUENCE, NB_TYPING_LIST) +
                                              const_name("[") +
                                              make_caster<moto::var>::Name +
                                              const_name(" | ") +
                                              make_caster<moto::func>::Name +
                                              const_name(" | moto.sym]"))

    list_caster<std::vector<handle>, handle> list_cast;

    bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) {
        if (!list_cast.from_python(src, flags, cleanup)) {
            return false;
        }
        auto &l = list_cast.value;
        value.reserve(l.size());
        value.clear();
        for (auto &ex : l) {
            auto type = ex.type();
            if (nb::isinstance<moto::shared_expr>(ex)) {
                value.push_back(nb::cast<moto::shared_expr &>(ex));
            } else if (nb::hasattr(ex, "sym_base")) {
                value.push_back(nb::cast<moto::var &>(ex.attr("sym_base")));
            } else {
                nb::print("Unknown type in var_list caster: ", ex);
            }
        }
        return true;
    }
};
} // namespace detail
} // namespace nanobind
