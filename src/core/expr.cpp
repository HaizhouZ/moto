#include <condition_variable>
#include <iostream>
#include <moto/core/expr.hpp>

namespace moto {
INIT_UID_(expr);

void expr::async_ready_status::set_ready_status(bool ready) {
    std::lock_guard<std::mutex> lock(ready_mutex_);
    ready_ = ready;
    ready_cond_.notify_all();
}
bool expr::async_ready_status::wait_until_ready() {
    std::unique_lock<std::mutex> lock(ready_mutex_);
    ready_cond_.wait(lock, [&, this]() {
        return ready_ != utils::optional_bool::Unset;
    });            ///< wait until the ready state is set
    return ready_; // return the ready state
}

expr::expr(const std::string &name, size_t dim, field_t field) {
    name_ = name;
    dim_ = dim;
    tdim_ = dim; // by default tdim = dim, can be changed in derived classes
    field_ = field;
    uid_.set_inc(); ///< set a new uid
}
expr::expr(const expr &rhs)
    : name_(rhs.name_), dim_(rhs.dim_), tdim_(rhs.tdim_), field_(rhs.field_),
      uid_(rhs.uid_), finalized_(false), dep_(rhs.dep_) {
    fmt::print("Copying expr {} with uid {} to new uid {}\n", rhs.name_, rhs.uid_, uid_);
} ///< copy constructor

bool expr::finalize(bool block_until_ready) {
    // fmt::print("finalizing expr {} uid {} field {}\n", name_, uid_, field::name(field_));
    // fmt::print("dim {} finalized {}\n", dim_, finalized_);
    if (!*this) {
        throw std::runtime_error(fmt::format("cannot finalize null expr"));
    }
    for (auto &d : dep_) {
        if (!d->finalize(block_until_ready)) {
            throw std::runtime_error(fmt::format("cannot finalize dependency expr {} uid {} of expr {} uid {}", d->name(), d->uid(), name_, uid_));
        }
    }
    if (!finalized()) {
        finalize_impl();
        finalized_ = (field_ != __undefined);
        if (block_until_ready)
            wait_until_ready(); ///< wait until the expression is ready
    }
    return finalized();
}
void expr::set_ready_status(bool ready) {
    async_ready_status_.set_ready_status(ready);
}

bool expr::wait_until_ready() const {
    if (!finalized_) {
        throw std::runtime_error(fmt::format("Expression {} with uid {} is not finalized, cannot call wait_until_ready", name_, uid_));
    }
    return async_ready_status_.wait_until_ready(); // return the ready state
}

expr_inarg_list::expr_inarg_list(const expr_list &exprs) {
    reserve(exprs.size());
    for (expr &ex : exprs) {
        emplace_back(ex);
    }
} ///< constructor from a vector of shared_expr

expr_list::expr_list(const expr_inarg_list &exprs) {
    reserve(exprs.size());
    for (const expr &ex : exprs) {
        emplace_back(ex);
    }
} ///< constructor from a vector of reference wrappers

} // namespace moto