#include <moto/core/expr.hpp>
namespace moto {
INIT_UID_(expr);
bool expr::finalize() {
    if (!finalized()) {
        shared_ ? shared_->finalize_impl() : finalize_impl();
        finalized() = (field() != __undefined);
    }
    return finalized();
}
} // namespace moto