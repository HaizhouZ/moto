#include <moto/core/external_function.hpp>
#include <moto/ocp/cost.hpp>

namespace moto {
void cost_impl::substitute_x_to_y() {
    for (auto &arg : in_args_) {
        switch (arg->field_) {
        case __x:
            fmt::print("substitution in cost {}: inarg {} with {}\n",
                       name_, arg->name_, arg->name_ + "_nxt");
            substitute(arg, expr_index::get<sym>(arg->name_ + "_nxt"));
            break;
        case __u:
            throw std::runtime_error(fmt::format(
                "cost {} cannot be terminal state-only cost, it has input arguments of type {}",
                name_, magic_enum::enum_name(arg->field_)));
        default:
            // do nothing
            break;
        }
    }
}
} // namespace moto