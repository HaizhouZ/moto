#include <moto/core/expr.hpp>

namespace moto {
namespace impl {
bool expr::finalize() {
    if (!finalized) {
        finalize_impl();
        finalized = (field_ != __undefined);
        expr_lookup::get(uid_).lock_index_by_name();
    }
    return finalized;
}
} // namespace impl
} // namespace moto