#include <moto/core/expr.hpp>

namespace moto {
void expr_impl::add_to_index() {
    try {
        auto [it, inserted] = expr_index::by_name_.try_emplace(name_, shared_from_this());
        if (!inserted) {
            throw std::runtime_error(
                fmt::format("expr name conflicts {} of uid {} with existing uid {}",
                            name_, uid_, it->second->uid_));
        }
    } catch (const std::bad_weak_ptr &ex) {
        throw std::runtime_error(fmt::format("expr {} not created from shared_ptr", name_));
    }
    expr_index::all_[uid_] = shared_from_this();
}
bool expr_impl::finalize() {
    if (!finalized) {
        finalize_impl();
        if (field_ == __undefined) {
            throw std::runtime_error(fmt::format("expr {} field type undefined after finalization", name_));
        }
        finalized = true;
        add_to_index();
    }
    return finalized;
}
} // namespace moto