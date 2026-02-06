#ifndef __EXPRESSION_BASE__
#define __EXPRESSION_BASE__

#include <condition_variable>
#include <moto/core/fields.hpp>
#include <moto/utils/optional_boolean.hpp>
#include <moto/utils/shared.hpp>
#include <moto/utils/unique_id.hpp>

namespace moto {
class expr; // forward declaration of expr
using shared_expr = utils::shared<expr>; ///< shared pointer type for expr

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
    const auto &mem_name() const { return mem_name##_; } \
    void __set_##mem_name(const decltype(mem_name##_) &value) { mem_name##_ = value; }\
    const auto &__get_##mem_name() const { return mem_name##_; }

class ocp;
/**
 * @brief general expression base class (now merged with impl)
 */
class expr : public std::enable_shared_from_this<expr>, public utils::clone_base<expr> {
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
        async_ready_status() = default;
        async_ready_status(const async_ready_status &) {}    ///< copy constructor
        async_ready_status(async_ready_status &&) = default; ///< move constructor
    };

    bool finalized_ = false;

    std::string name_;
    size_t dim_ = 0;
    size_t tdim_; ///< tangent space dimension, for manifolds
    utils::unique_id<expr> uid_;
    field_t field_ = __undefined;
    expr_list dep_; // now a direct member, not a pointer

    bool default_active_status_ = true; ///< default active status when added to an ocp (false means must be explicitly activated)

    mutable async_ready_status async_ready_status_; ///< async ready status, if any

    /// @brief finalize the expression, immediately set ready
    virtual void finalize_impl() { set_ready_status(true); }

    void set_ready_status(bool ready); ///< set the ready state and notify condition variable

    expr(const expr &rhs); ///< copy constructor

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

    friend bool operator ==(const expr &lhs, const expr &rhs) noexcept{
        return lhs.uid_ == rhs.uid_;
    } ///< equality operator by comparing uids

    DEF_DEFAULT_CLONE(expr)

};

std::string format_as(const expr &e); ///< format an expression as a string for debugging

/// @brief list of expressions, used for function arguments
struct expr_inarg_list : public std::vector<std::reference_wrapper<expr>> {
    using std::vector<std::reference_wrapper<expr>>::vector; ///< inherit constructors from std::vector
    /// constructor from a vector of shared_expr
    expr_inarg_list(const expr_list &exprs);
}; ///< list of expressions

} // namespace moto

#endif /*__EXPRESSION_BASE__*/