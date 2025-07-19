#include <moto/core/expr.hpp>

namespace moto {
expr_list::expr_list(std::initializer_list<impl::expr *> exprs) {
    for (auto expr : exprs) {
        emplace_back(impl::shared_handle<impl::expr>(expr));
    }
}
namespace impl {
expr::expr(const std::string &name, size_t dim, field_t field)
    : name_(name), dim_(dim), uid_(max_uid++), field_(field) { expr_lookup::all_.push_back(nullptr); }
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