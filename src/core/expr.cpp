#include <condition_variable>
#include <moto/core/expr.hpp>
#include <moto/utils/optional_boolean.hpp>
#include <shared_mutex>
#include <iostream>

namespace moto {
namespace impl {
class expr_async_ready_status {
  private:
    utils::optional_bool ready_ = utils::optional_bool::Unset; ///< ready state of the expression
    std::mutex ready_mutex_;                                   ///< mutex for ready state
    std::condition_variable ready_cond_;                       ///< condition variable for ready state
  public:
    void set_ready_status(bool ready = true) {
        std::lock_guard<std::mutex> lock(ready_mutex_);
        ready_ = ready;
        ready_cond_.notify_all();
    } ///< set the ready state and notify condition variable

    bool wait_until_ready() {
        std::unique_lock<std::mutex> lock(ready_mutex_);
        ready_cond_.wait(lock, [&, this]() {
            return ready_ != utils::optional_bool::Unset;
        });            ///< wait until the ready state is set
        return ready_; // return the ready state
    }
};
} // namespace impl

INIT_UID_(expr);
expr::expr(const std::string &name, size_t dim, field_t field) {
    name_ = name;
    dim_ = dim;
    field_ = field;
    async_ready_status_ = new impl::expr_async_ready_status(); ///< initialize async ready status
    uid_.set_inc();                                            ///< set a new uid
}
expr::expr(const expr &rhs)
    : name_(rhs.name_), dim_(rhs.dim_), field_(rhs.field_),
      uid_(rhs.uid_), finalized_(false), dep_(rhs.dep_) {
    async_ready_status_ = new impl::expr_async_ready_status(); ///< initialize async ready status
    fmt::print("Copying expr {} with uid {} to new uid {}\n", rhs.name_, rhs.uid_, uid_);
} ///< copy constructor

bool expr::finalize(bool block_until_ready) {
    // fmt::print("finalizing expr {} uid {} field {}\n", name_, uid_, field::name(field_));
    // fmt::print("dim {} finalized {}\n", dim_, finalized_);
    if (!finalized()) {
        finalize_impl();
        finalized_ = (field_ != __undefined);
        if (block_until_ready)
            wait_until_ready(); ///< wait until the expression is ready
    }
    return finalized();
}
void expr::set_ready_status(bool ready) {
    if (!async_ready_status_) {
        throw std::runtime_error(fmt::format("Expression {} with uid {} has no async ready status", name_, uid_));
    }
    async_ready_status_->set_ready_status(ready);
}

bool expr::wait_until_ready() const {
    if (!finalized_) {
        throw std::runtime_error(fmt::format("Expression {} with uid {} is not finalized", name_, uid_));
    }
    return async_ready_status_->wait_until_ready(); // return the ready state
}
expr::~expr() {
    if (async_ready_status_)
        delete async_ready_status_.get(); ///< delete the async ready status if it exists
}

} // namespace moto