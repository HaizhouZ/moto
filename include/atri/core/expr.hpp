#ifndef __EXPRESSION_BASE__
#define __EXPRESSION_BASE__

#include <atri/core/fields.hpp>
#include <casadi/casadi.hpp>

namespace atri {
class expr;
def_ptr(expr);
/**
 * @brief index of all syms
 *
 */
class expr_index {
    inline static std::vector<expr_ptr_t> all_;
    inline static std::unordered_map<std::string, expr_ptr_t> by_name_{};
    friend class expr;

  public:
    static const auto &get(const std::string &name) { return by_name_.at(name); }
    static const auto &get(size_t uid) { return all_.at(uid); }
};
/**
 * @brief general expression base class
 */
class expr : public std::enable_shared_from_this<expr> {
  private:
    static size_t max_uid; // uid used to index global expressions
    bool finalized = false;
    void add_to_index() {
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

  protected:
    virtual bool finalize_impl() { return true; }

  public:
    const std::string name_;
    const size_t dim_;
    const size_t uid_;
    const field_t field_;

    /**
     * @brief Construct a new expr
     * @note by default set field to field_t::NUM (i.e., undecided), also by default dim = 0
     * @param name name of the expression
     * @param dim dimension
     * @param field
     */
    expr(const std::string &name, size_t dim, field_t field)
        : name_(name), dim_(dim), uid_(max_uid++), field_(field) { expr_index::all_.push_back(nullptr); }

    expr(expr &&rhs)
        : name_(std::move(rhs.name_)), dim_(rhs.dim_), uid_(rhs.uid_), field_(rhs.field_) { expr_index::all_.push_back(nullptr); }

    auto make_vec(scalar_t *ptr) { return mapped_vector(ptr, dim_); }

    auto make_vec(const scalar_t *ptr) { return mapped_const_vector(ptr, dim_); }

    /**
     * @brief finalize this expression. Will be called upon added to a problem
     * @note derived classes
     * @retval true if successfully finalized
     */
    bool finalize() {
        if (!finalized) {
            finalized = finalize_impl();
            if (field_ == __undefined) {
                throw std::runtime_error(fmt::format("expr {} field type undefined", name_));
            }
            add_to_index(); // cannot be called from constructor, because shared_from_this() is not available yet
            // expr_index::by_uid_.try_emplace(uid_, shared_from_this());
        }
        return finalized;
    }
    /**
     * @brief get other variables related to this expression, by default will return empty
     * @return std::vector<expr_ptr_t> list of expressions
     */
    virtual std::vector<expr_ptr_t> get_aux() { return {}; }
};

namespace cs = casadi;

/**
 * @brief pointer wrapper of symbolic expressions like primal variables or parameters
 * @warning symbolic computation is via cs::SX, so dont cast expr_ptr_t back to sym! (unless you know it is safe)
 */
struct sym : public expr_ptr_t, public cs::SX {
    sym(const std::string &name, size_t dim, field_t type)
        : expr_ptr_t(new expr(name, dim, type)), cs::SX(cs::SX::sym(name, dim)) {
        assert(size_t(type) <= field::num_sym);
    }
    sym(const expr_ptr_t &rhs)
        : expr_ptr_t(rhs), cs::SX(cs::SX::sym(rhs->name_, rhs->dim_)) {}
    sym() = default;

    bool operator==(const sym &rhs) {
        return this->ptr() == rhs.ptr();
    }

    bool empty() { return this->ptr() == nullptr; }

    using expr_ptr_t::operator->;
    expr *ptr() const { return expr_ptr_t::get(); }
};

/**
 * @brief protected vector of expressions
 * @note when get() is called it must not be empty
 */
struct expr_list {
  private:
    std::vector<expr_ptr_t> expr_;

  protected:
    void add(std::initializer_list<expr_ptr_t> exprs) { expr_.insert(expr_.end(), exprs); }
    void add(expr_ptr_t exprs) { expr_.emplace_back(exprs); }

  public:
    expr_list(std::initializer_list<expr_ptr_t> exprs) : expr_(exprs) {}
    expr_list(std::initializer_list<expr *> exprs) {
        for (auto expr : exprs) {
            expr_.emplace_back(expr);
        }
    }
    expr_list() = default;
    const auto &get() const {
        assert(!expr_.empty());
        return expr_;
    }
};

} // namespace atri

#endif /*__EXPRESSION_BASE_*/