#include <condition_variable>
#include <iostream>
#include <moto/core/expr.hpp>
#include <unordered_map>

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
      uid_(rhs.uid_), finalized_(false), codegen_(rhs.codegen_), dep_(rhs.dep_) {
} ///< copy constructor

expr_handle expr::handle() {
    try {
        return shared_from_this();
    } catch (const std::bad_weak_ptr &) {
        throw std::runtime_error(fmt::format(
            "expression {} uid {} has no owning handle", name_, uid_));
    }
}
expr_handle expr::handle() const {
    try {
        return std::const_pointer_cast<expr>(shared_from_this());
    } catch (const std::bad_weak_ptr &) {
        throw std::runtime_error(fmt::format(
            "expression {} uid {} has no owning handle", name_, uid_));
    }
}

std::string format_as(const expr &e) {
    return fmt::format("expr(name={}, uid={}, dim={}, field={})", e.name(), e.uid(), e.dim(), e.field());
}

std::vector<expr *> expr::bind_codegen_graph(
    codegen_context_ptr context, std::span<expr *const> additional_roots) {
    context = resolve_codegen(context ? std::move(context) : codegen_);
    std::vector<std::pair<expr *, bool>> pending;
    for (auto *root : additional_roots)
        pending.emplace_back(root, false);
    pending.emplace_back(this, false);
    std::vector<expr *> graph;
    std::unordered_map<expr *, bool> finished;
    while (!pending.empty()) {
        const auto [current, exiting] = pending.back();
        pending.pop_back();
        if (exiting) {
            finished.at(current) = true;
            graph.push_back(current);
            continue;
        }
        const auto [it, inserted] = finished.emplace(current, false);
        if (!inserted) {
            if (!it->second)
                throw std::invalid_argument(fmt::format(
                    "cyclic expression dependency at {}", current->name()));
            continue;
        }
        if (!current || !*current)
            throw std::runtime_error("cannot finalize null expr");
        if (current->codegen_)
            current->codegen_->require_compatible(*context, current->name());
        pending.emplace_back(current, true);
        for (auto dependency = current->dep_.rbegin();
             dependency != current->dep_.rend(); ++dependency)
            pending.emplace_back(dependency->get(), false);
    }
    // Commit only after the entire dependency graph has passed validation.
    for (auto *current : graph)
        if (!current->codegen_)
            current->codegen_ = context;
    return graph;
}

void expr::bind_codegen(codegen_context_ptr context) {
    bind_codegen_graph(std::move(context));
}

bool expr::finalize(bool block_until_ready, codegen_context_ptr context) {
    // Revisit the current graph on every call: dependencies can be edited
    // between calls. Within this call, process each shared dependency once.
    for (auto *current : bind_codegen_graph(std::move(context))) {
        if (!current->finalized()) {
            current->finalize_impl();
            current->finalized_ = (current->field_ != __undefined);
        }
        if (current != this && !current->finalized())
            throw std::runtime_error(fmt::format(
                "cannot finalize dependency expr {} uid {} of expr {} uid {}",
                current->name(), current->uid(), name_, uid_));
        if (block_until_ready)
            current->wait_until_ready();
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
} ///< constructor from owning expression handles

expr_list::expr_list(const expr_inarg_list &exprs) {
    reserve(exprs.size());
    for (const expr &ex : exprs) {
        emplace_back(ex.handle());
    }
} ///< constructor from a vector of reference wrappers

} // namespace moto
