#ifndef __EXPRESSION_BASE__
#define __EXPRESSION_BASE__

#include <casadi/casadi.hpp>
#include <moto/core/fields.hpp>

namespace moto {
class expr_impl; // forward declaration of expr_impl
def_ptr_named(expr, expr_impl);
struct expr_list; // forward declaration of expr_list
/**
 * @brief a wrapper of std::vector<expr_ptr_t> to allow easy construction
 *
 */
struct expr_list : public std::vector<expr_ptr_t> {
    /**
     * @brief construct a new expr list object
     * will construct std::shared_ptr(expr)
     * @param exprs initializer list of raw expr pointers
     */
    expr_list(std::initializer_list<expr_impl *> exprs) {
        for (auto expr : exprs) {
            emplace_back(expr);
        }
    }
    /**
     * @brief extend the list with another list
     *
     * @param rhs lvalue ref, i.e., not movable, will be copied
     */
    void extend(const expr_list &rhs) {
        insert(end(), rhs.begin(), rhs.end());
    }
    /**
     * @brief extend the list with another list
     *
     * @param rhs rvalue ref, i.e., movable, will be moved
     */
    void extend(expr_list &&rhs) {
        insert(end(), std::make_move_iterator(rhs.begin()), std::make_move_iterator(rhs.end()));
    }
    using std::vector<expr_ptr_t>::vector; /// < inherit constructors
};
struct shared_base {
    virtual expr_impl &val() const = 0;
};

template <typename derived>
    requires std::derived_from<derived, expr_impl>
struct shared_ptr_ : public std::shared_ptr<derived>, public shared_base {
    using std::shared_ptr<derived>::shared_ptr;
    expr_impl &val() const override { return *(*this); }
};

def_ptr(shared_base);
/**
 * @brief index of all syms
 *
 */
struct expr_index {
    /// all expressions, indexed by uid, nullptr if not finalized (only placeholder)
    inline static std::vector<shared_base_ptr_t> all_;
    /// all expressions, indexed by name
    inline static std::unordered_map<std::string, shared_base_ptr_t> by_name_{};

    /// @brief get an expression by name
    template <typename T = shared_ptr_<expr_impl>,
              typename derived = std::remove_reference_t<decltype(*std::declval<T>())>>
        requires(std::is_base_of_v<shared_ptr_<derived>, T> && std::is_base_of_v<expr_impl, derived>)
    static const T &get(const std::string &name) { return static_cast<T &>(*by_name_.at(name)); }
    /// @brief get an expression by uid
    template <typename T = shared_ptr_<expr_impl>,
              typename derived = std::remove_reference_t<decltype(*std::declval<T>())>>
        requires(std::is_base_of_v<shared_ptr_<derived>, T> && std::is_base_of_v<expr_impl, derived>)
    static const auto &get(size_t uid) { return static_cast<T &>(*all_.at(uid)); }
};
constexpr size_t dim_tbd = 0;
/**
 * @brief general expression base class
 */
class expr_impl {
  private:
    inline static size_t max_uid = 0; /// < uid used to index global expressions
    bool finalized = false;

  protected:
    expr_list aux_;
    virtual void finalize_impl() {}

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
    expr_impl(const std::string &name, size_t dim, field_t field)
        : name_(name), dim_(dim), uid_(max_uid++), field_(field) { expr_index::all_.push_back(nullptr); }
    /**
     * @brief make a const vector from a pointer
     * @param ptr pointer to the data
     * @return mapped_const_vector
     */
    [[nodiscard]]
    auto make_vec(scalar_t *ptr) { return mapped_vector(ptr, dim_); }
    /**
     * @brief make a const vector from a pointer
     * @param ptr pointer to the data
     * @return mapped_const_vector
     */
    [[nodiscard]]
    auto make_vec(const scalar_t *ptr) { return mapped_const_vector(ptr, dim_); }

    /**
     * @brief finalize this expression. Will be called upon added to a problem
     * @note derived classes
     * @retval true if successfully finalized
     */
    bool finalize();
    /**
     * @brief get other variables related to this expression
     * @return pointer to std::vector<expr_ptr_t> list of expressions, default is nullptr
     */
    expr_list &get_aux() { return aux_; }
};
template <typename derived, typename derived_shared = shared_ptr_<derived>>
    requires std::derived_from<derived, expr_impl>
struct shared_ : public shared_ptr_<derived> {
    using shared_ptr = shared_ptr_<derived>;
    shared_(derived *p) : shared_ptr(p) { set_shared_from(p); }
    void reset(derived *p) {
        expr_index::all_[(*this)->uid_].reset(); // clear previous one
        expr_index::by_name_.erase((*this)->name_); // clear previous one
        shared_ptr::reset(p);
        set_shared_from(p);
    }
    void set_shared_from(derived *p) {
        const auto &cur = static_cast<derived_shared&>(*this);
        expr_index::all_[p->uid_].reset(new derived_shared(cur));
        auto [it, inserted] = expr_index::by_name_.try_emplace(p->name_, expr_index::all_[p->uid_]);
        if (!inserted and it->second->val().uid_ != p->uid_) {
            throw std::runtime_error(
                fmt::format("expr name conflicts {} of uid {} with existing uid {}",
                            p->name_, p->uid_, it->second->val().uid_));
        }
    }
    using shared_ptr::operator->;
    shared_() = default;
};
namespace cs = casadi;

/**
 * @brief pointer wrapper of symbolic expressions like primal variables or parameters
 * @warning symbolic computation is via cs::SX, so dont cast expr_ptr_t back to sym! (unless you know it is safe)
 */
struct sym : public cs::SX, public shared_<expr_impl, sym> {
    using base = shared_<expr_impl, sym>;
    /**
     * @brief Construct a new sym object
     *
     * @param name name of the symbolic variable
     * @param dim dimension of the symbolic variable
     * @param type type of the symbolic variable, must be one of the symbolic fields
     */
    sym(const std::string &name, size_t dim, field_t type)
        : base(new expr_impl(name, dim, type)), cs::SX(cs::SX::sym(name, dim)) {
        assert(size_t(type) <= field::num_sym || type == __usr_var);
    }
    sym() = default;
    /**
     * @brief check if this sym is equal to another sym
     * only check by pointer to the expression_impl, not by name or uid or cs::SX
     * @param rhs another sym to compare with
     */
    bool operator==(const sym &rhs) {
        return expr_ptr_t::get() == rhs.get();
    }

    using base::operator->; // inherit operator-> to access expr_impl methods
    using base::get;
};
} // namespace moto

#endif /*__EXPRESSION_BASE_*/