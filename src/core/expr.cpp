#include <moto/core/expr.hpp>
namespace moto {
INIT_UID_(expr);
bool expr::finalize() {
    fmt::print("finalizing expr {} uid {} field {}\n", name_, uid_, field::name(field_));
    fmt::print("dim {} finalized {}\n", dim_, finalized_);
    if (!finalized()) {
        finalize_impl();
        finalized_ = (field() != __undefined);
    }
    return finalized();
}
} // namespace moto