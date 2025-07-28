#include <moto/core/external_function.hpp>
#include <moto/ocp/cost.hpp>

namespace moto {
void generic_cost::finalize_impl() {
    if (finalize_hint_.substitute_x_to_y) {
        for (sym &arg : in_args_) {
            switch (arg.field()) {
            case __x:
                fmt::print("substitution in cost {}: inarg {} with {}\n",
                           name_, arg.name(), arg.name() + "_nxt");
                substitute(arg, arg.next());
                break;
            case __u:
                throw std::runtime_error(fmt::format(
                    "cost {} can only be terminal state-only cost, but has input arguments of type {}",
                    name_, field::name(arg.field())));
            default:
                // do nothing
                break;
            }
        }
    }
    // fmt::print("field_hint for cost {} is {}\n", name_, finalize_hint_.substitute_x_to_y);
    // finalize the base class
    generic_func::finalize_impl();
    return;
}

cost::cost(const std::string &name, approx_order order)
    : func(generic_cost(name, order, 1, __cost)) {}

cost::cost(const std::string &name, const var_inarg_list &in_args, const cs::SX &out, approx_order order)
    : func(generic_cost(name, in_args, out, order, __cost)) {
    assert(out.is_scalar() && "cost output must be a scalar");
}

cost &cost::as_terminal() {
    (*this)->name_ += "_terminal";
    (*this)->finalize_hint_.substitute_x_to_y = true;
    return (*this);
}
} // namespace moto