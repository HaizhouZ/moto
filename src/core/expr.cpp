#include <moto/core/expr.hpp>
namespace moto {
INIT_UID_(expr::impl);
bool expr::finalize() {
    if (!finalized()) {
        impl_->finalize_impl();
        impl_->finalized_ = (field() != __undefined);
    }
    return finalized();
}
} // namespace moto