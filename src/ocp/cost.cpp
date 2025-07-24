#include <moto/core/external_function.hpp>
#include <moto/ocp/cost.hpp>

namespace moto {
void cost::finalize_impl() {
    if (finalize_hint().substitute_x_to_y) {
        for (sym &arg : in_args()) {
            switch (arg.field()) {
            case __x:
                fmt::print("substitution in cost {}: inarg {} with {}\n",
                           name(), arg.name(), arg.name() + "_nxt");
                get_impl().substitute(arg, arg.next());
                break;
            case __u:
                throw std::runtime_error(fmt::format(
                    "cost {} can only be terminal state-only cost, but has input arguments of type {}",
                    name(), field::name(arg.field())));
            default:
                // do nothing
                break;
            }
        }
    }
    // finalize the base class
    func::finalize_impl();
    return;
}
} // namespace moto