#ifndef __EXPRESSION_BASE__
#define __EXPRESSION_BASE__

#include <moto/core/fields.hpp>

namespace moto {
class expr; // forward declaration of expr
/**
 * @brief Reference wrapper for moto::expr
 * @details this class maintains shared ownership of the expression
 *
 * The goal is to provide a lightweight reference to an expression that can be shared across different parts of the code.
 * It allows for easy passing of expressions when writing / storage / reference is required.
 */
struct expr_ref {
  private:
    std::shared_ptr<expr> expr_ = nullptr; ///< unique pointer to the expression

  public:
    template <typename T>
        requires std::is_base_of_v<expr, std::remove_reference_t<T>>
    expr_ref(T &&e) {
        using T_ = std::remove_cvref_t<T>;
        T_ *ptr = const_cast<T_ *>(&e); // get the pointer to the expression
        assert(!(e.shared_ && e.is_shared_) && "expr is shared, but has shared ref");
        if (e.shared_) {                                // if holds a shared ref
            expr_ = e.shared_.expr_;                    // use the shared ref
        } else if (e.is_shared_) {                      // if the expression is already shared
            expr_ = ptr->shared_from_this();            // use the shared pointer
        } else {                                        // new obj not shared
            expr_ = std::make_shared<T_>(std::move(e)); // move the expression
            ptr->set_shared(*this);                     // set the shared pointer to this
            e.shared_->is_shared_ = true;               // mark it as shared
        }
    }
    template <typename T>
        requires std::is_base_of_v<expr, std::remove_reference_t<T>>
    operator T &() const {
        if (!expr_) {
            throw std::runtime_error("expr_ref is null");
        }
        return static_cast<T &>(*expr_);
    }

    expr &operator*() const {
        assert(expr_ != nullptr && "dereference of null expr_ref");
        return *expr_;
    }

    expr *operator->() const {
        assert(expr_ != nullptr && "dereference of null expr_ref");
        return expr_.get();
    }
    operator bool() const { return expr_.operator bool(); }
    expr_ref() {}
};
/// @brief type alias for a list of expression reference
using expr_list = std::vector<expr_ref>;

constexpr size_t dim_tbd = 0;

#define CONST_ATTR_GETTER(mem_name) \
    const auto &mem_name() const { return mem_name##_; }

#define ATTR_GETTER(mem_name)                \
    auto &mem_name() { return mem_name##_; } \
    const auto &mem_name() const { return mem_name##_; }

#define SHARED_ATTR_GETTER(mem_name, derived)                                               \
    auto &mem_name() { return shared_ ? shared_->as<derived>().mem_name##_ : mem_name##_; } \
    const auto &mem_name() const { return shared_ ? shared_->as<derived>().mem_name##_ : mem_name##_; }

#define INIT_UID_(type) size_t type::max_uid = 0;
constexpr size_t uid_max = std::numeric_limits<size_t>::max();
/**
 * @brief general expression base class
 */
class expr : public std::enable_shared_from_this<expr> {
  private:
    static size_t max_uid; /// < uid used to index global expressions
    bool finalized_ = false;

    std::string name_;            ///< name of the expression
    size_t dim_ = 0;              ///< dimension of the expression, 0 if not set
    size_t uid_ = uid_max;        ///< unique id of the expression, used to index in expr_lookup
    field_t field_ = __undefined; ///< field of the expression, default is field_t::__undefined (i.e., undecided)
    expr_list dep_;               ///< dependencies of the expression, i.e., other expressions that this expression depends on

  protected:
    expr_ref shared_;        ///< shared reference for shared attributes
    bool is_shared_ = false; ///< denotes if this expression is shared
    void set_shared(const expr_ref &e) { shared_ = e; }

    friend class expr_ref; ///< allow expr_ref to access private members

    virtual void finalize_impl() {}

  public:
    SHARED_ATTR_GETTER(name, expr);      ///< getter for name
    SHARED_ATTR_GETTER(dim, expr);       ///< getter for dim
    SHARED_ATTR_GETTER(uid, expr);       ///< getter for uid
    SHARED_ATTR_GETTER(field, expr);     ///< getter for field
    SHARED_ATTR_GETTER(finalized, expr); ///< getter for finalized

    CONST_ATTR_GETTER(is_shared); ///< getter for shared status

    template <typename T>
        requires std::is_base_of_v<expr, std::remove_reference_t<T>>
    void add_dep(T &&e) {
        assert(static_cast<const expr &>(e) && "cannot add expr dependency to a null expression");
        dep_.emplace_back(std::forward<T>(e));
    }

    operator bool() const { return uid() < uid_max; }

    ATTR_GETTER(shared); ///< getter for shared

    expr() = default; // default constructor for derived classes
    /**
     * @brief Construct a new expr
     * @note by default set field to field_t::NUM (i.e., undecided), also by default dim = 0
     * @param name name of the expression
     * @param dim dimension
     * @param field
     */
    expr(const std::string &name, size_t dim, field_t field)
        : name_(name), dim_(dim), field_(field), uid_(max_uid++) {}
    /// copy constructor will get a new uid and leave un-finalized
    expr(const expr &rhs)
        : name_(rhs.name_), dim_(rhs.dim_), field_(rhs.field_), uid_(max_uid++),
          dep_(rhs.dep_), shared_(rhs.shared_) {}
    /// move constructor
    expr(expr &&rhs)
        : name_(std::move(rhs.name_)), dim_(rhs.dim_), field_(rhs.field_), uid_(rhs.uid_),
          dep_(std::move(rhs.dep_)), shared_(std::move(rhs.shared_)), finalized_(rhs.finalized_) {
        rhs.uid_ = uid_max; // reset the uid of the moved object
    }

    expr &operator=(const expr &) = default;
    expr &operator=(expr &&) = default;
    /**
     * @brief make a const vector from a pointer
     * @param ptr pointer to the data
     * @return mapped_const_vector
     */
    [[nodiscard]]
    auto make_vec(scalar_t *ptr) const { return mapped_vector(ptr, dim_); }
    /**
     * @brief make a const vector from a pointer
     * @param ptr pointer to the data
     * @return mapped_const_vector
     */
    [[nodiscard]]
    auto make_vec(const scalar_t *ptr) const { return mapped_const_vector(ptr, dim_); }

    /**
     * @brief finalize this expression. Will be called upon added to a problem
     * @warning if one is finalized before dependencies, failure may happen
     * @retval true if successfully finalized
     */
    bool finalize();
    /**
     * @brief get other variables related to this expression
     */
    CONST_ATTR_GETTER(dep); ///< getter for dep

    template <typename T>
        requires std::is_base_of_v<expr, T>
    T &as() { return static_cast<T &>(*this); }
};
} // namespace moto

#endif /*__EXPRESSION_BASE_*/