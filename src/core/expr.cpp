#include <moto/core/expr.hpp>
namespace moto {
INIT_UID_(expr);
bool expr::finalize() {
    if (!finalized()) {
        finalize_impl();
        finalized_ = (field() != __undefined);
    }
    return finalized();
}
} // namespace moto