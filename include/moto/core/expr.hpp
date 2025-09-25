#ifndef __EXPRESSION_BASE__
#define __EXPRESSION_BASE__

#include <condition_variable>
#include <moto/core/fields.hpp>
#include <moto/utils/optional_boolean.hpp>
#include <moto/utils/unique_id.hpp>

namespace moto {
class expr; // forward declaration of expr
/**
 * @brief shared reference wrapper of expressions
 * @note can be converted to expr (and its derived types) references
 */
class shared_expr {
  private:
    using ptr_t = std::shared_ptr<expr>;
    ptr_t ptr_;

  public:
    shared_expr() = default; ///< default constructor
    template <typename U>
    void reset(U *u) { ptr_.reset(u); } ///< reset the pointer
    template <typename U, typename U_ = std::remove_cvref_t<U>>
        requires std::is_base_of_v<expr, U_>
    shared_expr(U &&u) {
        if (u.use_count() > 0) {
            *this = const_cast<U_ &&>(u).get_shared();
        } else {
            ptr_.reset(new U_(std::forward<U>(u)));
            if (u) {
                const_cast<U_ &>(u).set_shared(shared_expr(*this));
            }
        }
    } ///< constructor from a reference
    shared_expr(ptr_t &&p) : ptr_(std::move(p)) {}                ///< constructor from a shared pointer
    shared_expr(const ptr_t &p) : ptr_(p) {}                      ///< constructor from a shared pointer
    shared_expr(const shared_expr &rhs) = default;                ///< copy constructor
    shared_expr(shared_expr &&rhs) noexcept = default;            ///< move constructor
    shared_expr &operator=(const shared_expr &rhs) = default;     ///< copy assignment operator
    shared_expr &operator=(shared_expr &&rhs) noexcept = default; ///< move assignment operator
    shared_expr(const std::reference_wrapper<expr> &rhs)
        : shared_expr((const expr &)rhs) {} ///< constructor from a reference wrapper

    template <typename U, typename U_ = std::remove_cvref_t<U>>
        requires std::is_base_of_v<expr, U_> || std::is_base_of_v<U_, expr>
    operator U &() const { return static_cast<U &>(*ptr_); } ///< dereference operator for derived types

    expr *operator->() const { return ptr_.get(); }

    explicit operator bool() const { return static_cast<bool>(ptr_); } ///< conversion to bool operator
    size_t use_count() const { return ptr_.use_count(); }              ///< get the use count of the pointer
    virtual ~shared_expr();                                            ///< virtual destructor
};
inline bool operator==(const shared_expr &lhs, const shared_expr &rhs) {
    return lhs.operator->() == rhs.operator->();
} ///< equality operator
template <typename U>
    requires std::is_base_of_v<expr, std::remove_cvref_t<U>>
inline bool operator==(const shared_expr &lhs, const U &rhs) {
    return lhs.operator->() == static_cast<const expr *>(&rhs);
} ///< equality operator for shared_expr and derived types
struct expr_inarg_list; ///< forward declaration
/// @brief list of expressions, used for storing expressions in a vector
struct expr_list : public std::vector<shared_expr> {
    using std::vector<shared_expr>::vector; ///< inherit constructors from std::vector
    /// constructor from a vector of reference wrappers
    expr_list(const expr_inarg_list &exprs);
};

constexpr size_t dim_tbd = 0;

#define CONST_PROPERTY(mem_name) \
    const auto &mem_name() const { return mem_name##_; }

#define PROPERTY(mem_name)                   \
    auto &mem_name() { return mem_name##_; } \
    const auto &mem_name() const { return mem_name##_; }

class ocp;
/**
 * @brief general expression base class (now merged with impl)
 */
class expr : public std::enable_shared_from_this<expr> {
  public:
    static size_t max_uid; /// < uid used to index global expressions

  protected:
    class async_ready_status {
      private:
        utils::optional_bool ready_ = utils::optional_bool::Unset; ///< ready state of the expression
        std::mutex ready_mutex_;                                   ///< mutex for ready state
        std::condition_variable ready_cond_;                       ///< condition variable for ready state
      public:
        void set_ready_status(bool ready = true);
        bool wait_until_ready();
    };

    bool finalized_ = false;

    std::string name_;
    size_t dim_ = 0;
    size_t tdim_; ///< tangent space dimension, for manifolds
    utils::unique_id<expr> uid_;
    field_t field_ = __undefined;
    expr_list dep_; // now a direct member, not a pointer

    bool default_active_status_ = true; ///< default active status when added to an ocp (false means must be explicitly activated)

    mutable std::shared_ptr<async_ready_status> async_ready_status_; ///< async ready status, if any

    shared_expr shared_;

    /// @brief finalize the expression, immediately set ready
    virtual void finalize_impl() { set_ready_status(true); }

    void set_ready_status(bool ready); ///< set the ready state and notify condition variable

    expr(const expr &rhs); ///< copy constructor

    friend class shared_expr;

  public:
    PROPERTY(name);                 ///< getter for name
    PROPERTY(dim);                  ///< getter for dim
    PROPERTY(uid);                  ///< getter for uid
    PROPERTY(field);                ///< getter for field
    PROPERTY(finalized);            ///< getter for finalized
    PROPERTY(tdim)                  ///< tangent space dimension of the symbolic variable
    PROPERTY(default_active_status) ///< default active status when added to an ocp

    auto &dep() { return dep_; } ///< get the dependencies of this expression

    template <typename T>
        requires std::is_base_of_v<shared_expr, std::remove_reference_t<T>>
    void add_dep(T &&e) {
        assert(static_cast<const expr &>(e) && "cannot add expr dependency to a null expression");
        dep_.emplace_back(std::forward<T>(e));
    }

    template <typename T>
    void add_deps(const std::vector<T> &es) {
        for (const auto &e : es) {
            add_dep(e);
        }
    }

    virtual void add_to_ocp_callback(ocp *) {} /// callback when added to an ocp

    auto get_shared() {
        if (use_count() > 0)
            return shared_expr(shared_from_this());
        else if (shared_)
            return shared_;
        else
            throw std::runtime_error(fmt::format("Expression {} in field {} of uid {} is not shared",
                                                 name_, field::name(field_), uid_));
    }

    void set_shared(const shared_expr &ref) {
        assert(ref != *this && "cannot set shared pointer to self or null");
        shared_ = ref;
    } ///< set the shared pointer

    void set_shared(shared_expr &&ref) {
        assert(ref != *this && "cannot set shared pointer to self or null");
        shared_ = std::move(ref);
    } ///< set the shared pointer

    explicit operator bool() const { return uid_.is_valid(); }

    expr() = default; // default constructor for derived classes
    /**
     * @brief Construct a new expr
     * @note by default set field to field_t::NUM (i.e., undecided), also by default dim = 0
     * @param name name of the expression
     * @param dim dimension
     * @param field
     */
    expr(const std::string &name, size_t dim, field_t field);

    // Copy assignment operator (gets a new uid and leaves un-finalized)
    expr &operator=(const expr &rhs) = default;
    expr &operator=(expr &&rhs) noexcept = default;
    // Move constructor
    expr(expr &&rhs) = default;

    [[nodiscard]]
    auto make_vec(scalar_t *ptr) const { return mapped_vector(ptr, dim_); }
    [[nodiscard]]
    auto make_vec(const scalar_t *ptr) const { return mapped_const_vector(ptr, dim_); }

    bool finalize(bool block_until_ready = false); ///< finalize the expression, set ready status

    virtual bool wait_until_ready() const; ///< wait until the expression is ready

    size_t use_count() const { return shared_ ? shared_->use_count() : weak_from_this().use_count(); } ///< get the use count of the expression
};
/// @brief list of expressions, used for function arguments
struct expr_inarg_list : public std::vector<std::reference_wrapper<expr>> {
    using std::vector<std::reference_wrapper<expr>>::vector; ///< inherit constructors from std::vector
    /// constructor from a vector of shared_expr
    expr_inarg_list(const expr_list &exprs);
}; ///< list of expressions

} // namespace moto

#endif /*__EXPRESSION_BASE__*/