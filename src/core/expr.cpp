#include <moto/core/expr.hpp>

namespace moto {
namespace impl {
bool expr::finalize() {
    if (!finalized) {
        finalize_impl();
        finalized = (field_ != __undefined);
        expr_lookup::get(uid_).finalize_index_by_name(this);
    }
    return finalized;
}
} // namespace impl
} // namespace moto