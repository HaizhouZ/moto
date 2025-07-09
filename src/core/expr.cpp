#include <moto/core/expr.hpp>

namespace moto {
bool expr_impl::finalize() {
    if (!finalized) {
        finalize_impl();
        finalized = (field_ != __undefined);
        expr_index::get(uid_).finalize(this);
    }
    return finalized;
}
} // namespace moto