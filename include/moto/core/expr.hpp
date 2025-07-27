#ifndef __EXPRESSION_BASE__
#define __EXPRESSION_BASE__

#include <moto/core/fields.hpp>

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

    template <typename U, typename U_ = std::remove_cvref_t<U>>
        requires std::is_base_of_v<expr, U_> || std::is_base_of_v<U_, expr>
    operator U &() const { return static_cast<U &>(*ptr_); } ///< dereference operator for derived types

    expr *operator->() const { return ptr_.get(); }

    bool operator==(const shared_expr &rhs) const {
        return ptr_ == rhs.ptr_;
    } ///< equality operator
    template <typename U>
        requires std::is_base_of_v<expr, std::remove_cvref_t<U>>
    bool operator==(const U &rhs) const { return ptr_.get() == &rhs; }

    explicit operator bool() const { return static_cast<bool>(ptr_); } ///< conversion to bool operator
    size_t use_count() const { return ptr_.use_count(); }     ///< get the use count of the pointer
    virtual ~shared_expr(); ///< virtual destructor
};
/// @brief list of expressions, used for storing expressions in a vector
struct expr_list : public std::vector<shared_expr> {
    using std::vector<shared_expr>::vector; ///< inherit constructors from std::vector
};

constexpr size_t dim_tbd = 0;

#define CONST_PROPERTY(mem_name) \
    const auto &mem_name() const { return mem_name##_; }

#define PROPERTY(mem_name)                   \
    auto &mem_name() { return mem_name##_; } \
    const auto &mem_name() const { return mem_name##_; }

#define INIT_UID_(type) size_t type::max_uid = 0;

constexpr size_t uid_max = std::numeric_limits<size_t>::max();
/**
 * @brief general expression base class (now merged with impl)
 */
class expr : public std::enable_shared_from_this<expr> {
  public:
    static size_t max_uid; /// < uid used to index global expressions

  protected:
    bool finalized_ = false;
    std::string name_;
    size_t dim_ = 0;
    size_t uid_ = uid_max;
    field_t field_ = __undefined;
    expr_list dep_; // now a direct member, not a pointer

    virtual void finalize_impl() {}

    shared_expr shared_;

  public:
    PROPERTY(name);      ///< getter for name
    PROPERTY(dim);       ///< getter for dim
    PROPERTY(uid);       ///< getter for uid
    PROPERTY(field);     ///< getter for field
    PROPERTY(finalized); ///< getter for finalized

    auto &dep() { return dep_; } ///< get the dependencies of this expression

    template <typename T>
        requires std::is_base_of_v<shared_expr, std::remove_reference_t<T>>
    void add_dep(T &&e) {
        assert(static_cast<const expr &>(e) && "cannot add expr dependency to a null expression");
        dep_.emplace_back(std::forward<T>(e));
    }

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

    explicit operator bool() const { return uid_ < uid_max; }

    expr() = default; // default constructor for derived classes
    /**
     * @brief Construct a new expr
     * @note by default set field to field_t::NUM (i.e., undecided), also by default dim = 0
     * @param name name of the expression
     * @param dim dimension
     * @param field
     */
    expr(const std::string &name, size_t dim, field_t field) {
        name_ = name;
        dim_ = dim;
        field_ = field;
        uid_ = max_uid++;
    }

    // Copy assignment operator (gets a new uid and leaves un-finalized)
    expr &operator=(const expr &rhs) = default;
    // Move constructor
    expr(expr &&rhs)
        : name_(std::move(rhs.name_)), dim_(rhs.dim_), field_(rhs.field_),
          uid_(rhs.uid_), finalized_(rhs.finalized_), dep_(std::move(rhs.dep_)) {
        rhs.uid_ = uid_max;
    }
    expr(const expr &rhs)
        : name_(rhs.name_), dim_(rhs.dim_), field_(rhs.field_),
          uid_(max_uid++), finalized_(rhs.finalized_), dep_(rhs.dep_) {
        fmt::print("Copying expr {} with uid {} to new uid {}\n", rhs.name_, rhs.uid_, uid_);
    } ///< copy constructor

    virtual ~expr() = default;

    void set_impl(expr *e) {
        // No-op in merged version, kept for compatibility
    }

    [[nodiscard]]
    auto make_vec(scalar_t *ptr) const { return mapped_vector(ptr, dim_); }
    [[nodiscard]]
    auto make_vec(const scalar_t *ptr) const { return mapped_const_vector(ptr, dim_); }

    bool finalize();

    size_t use_count() const { return weak_from_this().use_count(); } ///< get the use count of the expression
};
/// @brief list of expressions, used for function arguments
struct expr_inarg_list : public std::vector<std::reference_wrapper<expr>> {
    using std::vector<std::reference_wrapper<expr>>::vector; ///< inherit constructors from std::vector
    expr_inarg_list(const expr_list &exprs) {                ///< constructor from a vector of shared_expr
        reserve(exprs.size());
        for (expr &ex : exprs) {
            emplace_back(ex);
        }
    } ///< constructor from a vector of shared_expr
}; ///< list of expressions

inline shared_expr::~shared_expr() {
    if (ptr_ && *ptr_)
        fmt::print("Shared object {} with uid {} is destroyed\n", ptr_->name(), ptr_->uid());
} ///< destructor

} // namespace moto

#endif /*__EXPRESSION_BASE__*/