#include <moto/core/external_function.hpp>
#include <moto/ocp/cost.hpp>

namespace moto {
namespace impl {
void cost::finalize_impl() {
    if (hint_.substitute_x_to_y) {
        for (auto &arg : in_args_) {
            switch (arg->field_) {
            case __x:
                fmt::print("substitution in cost {}: inarg {} with {}\n",
                           name_, arg->name_, arg->name_ + "_nxt");
                substitute(arg, expr_lookup::get<sym>(arg->uid_ + 1));
                break;
            case __u:
                throw std::runtime_error(fmt::format(
                    "cost {} can only be terminal state-only cost, but has input arguments of type {}",
                    name_, magic_enum::enum_name(arg->field_)));
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
} // namespace impl
} // namespace moto