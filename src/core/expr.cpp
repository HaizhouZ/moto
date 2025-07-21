#include <moto/core/expr.hpp>
namespace moto {
std::vector<expr_lookup::expr_handle> expr_lookup::all_{};                           ///< all expressions, indexed by uid, nullptr if not finalized (only placeholder)
std::unordered_map<std::string, expr_lookup::expr_handle *> expr_lookup::by_name_{}; ///< all expressions, indexed by name
INIT_UID_(expr);
expr *expr::create(const std::string &name, size_t dim, field_t field) {
    auto new_ = new expr();
    new_->name_ = name;
    new_->dim_ = dim;
    new_->field_ = field;
    new_->hd_ = &expr_lookup::all_.emplace_back(new_);
    auto [it, inserted] = expr_lookup::by_name_.try_emplace(name, new_->hd_);
    if (!inserted)
        throw std::runtime_error(fmt::format("expression name {} already exists with uid {}, current uid {}", name, (*it->second)->uid(), new_->uid_));
    return new_;
}
void expr::update_name(const std::string &name) {
    auto [it, inserted] = expr_lookup::by_name_.try_emplace(name, hd_);
    expr_lookup::by_name_.erase(name_); // remove old name
    if (!inserted) {
        throw std::runtime_error(fmt::format("expression name {} already exists with uid {}, current uid {}", name, (*it->second)->uid(), uid_));
    }
    name_ = name;
}
bool expr::finalize() {
    if (!finalized) {
        finalize_impl();
        finalized = (field_ != __undefined);
        // expr_lookup::get(uid_).lock_index_by_name();
    }
    return finalized;
}
} // namespace moto