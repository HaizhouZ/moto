#include <moto/core/external_function.hpp>
#include <moto/ocp/cost.hpp>

namespace moto {
void cost::impl::finalize_impl() {
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
    // finalize the base class
    func::impl::finalize_impl();
    return;
}
} // namespace moto