#include <moto/core/expr.hpp>
namespace moto {
INIT_UID_(expr::impl);
bool expr::finalize() {
    if (!finalized()) {
        finalize_impl();
        shared_->finalized_ = (field() != __undefined);
    }
    return finalized();
}
} // namespace moto