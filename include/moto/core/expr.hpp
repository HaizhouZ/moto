#ifndef __EXPRESSION_BASE__
#define __EXPRESSION_BASE__

#include <moto/core/fields.hpp>

namespace moto {
class expr; // forward declaration of expr
/**
 * @brief shared pointer to expr wrapper, used for sharing the wrapper
 * @note can be converted to expr reference
 */
template <typename T>
class shared_object {
  private:
    using ptr_t = std::shared_ptr<T>;
    ptr_t ptr_;

  public:
    shared_object() = default; ///< default constructor
    template <typename U>
    void reset(U *u) { ptr_.reset(u); } ///< reset the pointer
    template <typename U, typename U_ = std::remove_cvref_t<U>>
        requires std::is_base_of_v<T, U_>
    shared_object(U &&u) {
        ptr_.reset(new U_(std::forward<U>(u)));
    } ///< constructor from a reference
    template <typename U>
        requires std::is_base_of_v<T, std::remove_cvref_t<U>>
    operator U &() const { return static_cast<U &>(*ptr_); } ///< dereference operator
    T *operator->() const { return ptr_.get(); }             ///< arrow operator

    operator bool() const { return static_cast<bool>(ptr_); } ///< conversion to bool operator
    size_t use_count() const { return ptr_.use_count(); } ///< get the use count of the pointer
};

using shared_expr = shared_object<expr>; ///< type alias for shared expression reference
using expr_list = std::vector<shared_expr>;

constexpr size_t dim_tbd = 0;

#define CONST_ATTR_GETTER(mem_name) \
    const auto &mem_name() const { return mem_name##_; }

#define ATTR_GETTER(mem_name)                \
    auto &mem_name() { return mem_name##_; } \
    const auto &mem_name() const { return mem_name##_; }

#define IMPL_ATTR_GETTER(mem_name, derived)                              \
  public:                                                                \
    auto &mem_name() { return static_cast<impl &>(*impl_).mem_name##_; } \
    const auto &mem_name() const { return static_cast<const impl &>(*impl_).mem_name##_; }

#define DEF_IMPL_GETTER()                   \
  public:                                   \
    auto &get_impl() const {                \
        return static_cast<impl &>(*impl_); \
    } ///< get the shared pointer as impl

#define INIT_UID_(type) size_t type::max_uid = 0;
constexpr size_t uid_max = std::numeric_limits<size_t>::max();
/**
 * @brief general expression base class
 */
class expr {
  public:
    struct impl : public std::enable_shared_from_this<impl> {
        static size_t max_uid; /// < uid used to index global expressions
        bool finalized_ = false;

        std::string name_;            ///< name of the expression
        size_t dim_ = 0;              ///< dimension of the expression, 0 if not set
        size_t uid_ = uid_max;        ///< unique id of the expression, used to index in expr_lookup
        field_t field_ = __undefined; ///< field of the expression, default is field_t::__undefined (i.e., undecided)

        impl() = default;
        /// copy constructor will get a new uid and leave un-finalized
        impl &operator=(const impl &rhs) {
            name_ = rhs.name_;
            dim_ = rhs.dim_;
            field_ = rhs.field_;
            uid_ = max_uid++;
            finalized_ = false; // reset finalized
            return *this;
        } ///< copy assignment operator
        /// move constructor
        impl(impl &&rhs)
            : name_(std::move(rhs.name_)), dim_(rhs.dim_), field_(rhs.field_),
              uid_(rhs.uid_), finalized_(rhs.finalized_) {
            rhs.uid_ = uid_max; // reset the uid of the moved object
        }
        virtual ~impl() = default; ///< virtual destructor
    };
    virtual void finalize_impl() {}

    std::shared_ptr<expr::impl> impl_; ///< shared pointer to the expression, used for shared attributes

    IMPL_ATTR_GETTER(name, expr);      ///< getter for name
    IMPL_ATTR_GETTER(dim, expr);       ///< getter for dim
    IMPL_ATTR_GETTER(uid, expr);       ///< getter for uid
    IMPL_ATTR_GETTER(field, expr);     ///< getter for field
    IMPL_ATTR_GETTER(finalized, expr); ///< getter for finalized

    DEF_IMPL_GETTER();

  private:
    std::shared_ptr<expr_list> dep_; // cannot be in impl (avoid reference cycle)
  public:
    auto &dep() {
        if (!dep_) {
            static expr_list empty_list;
            return empty_list;
        }
        return *dep_;
    } ///< get the dependencies of this expression

    template <typename T>
        requires std::is_base_of_v<expr, std::remove_reference_t<T>>
    void add_dep(T &&e) {
        assert(static_cast<const expr &>(e) && "cannot add expr dependency to a null expression");
        if (!dep_) {
            dep_ = std::make_shared<expr_list>();
        }
        dep_->emplace_back(std::forward<T>(e));
    }

    explicit operator bool() const { return uid() < uid_max; }

    expr() = default; // default constructor for derived classes
    /**
     * @brief Construct a new expr
     * @note by default set field to field_t::NUM (i.e., undecided), also by default dim = 0
     * @param name name of the expression
     * @param dim dimension
     * @param field
     */
    expr(const std::string &name, size_t dim, field_t field) {
        impl_ = std::make_shared<impl>();
        impl_->name_ = name;
        impl_->dim_ = dim;
        impl_->field_ = field;
        impl_->uid_ = impl::max_uid++;
    }

    void set_impl(impl *impl) {
        impl_.reset(impl);
    } ///< set the implementation of the expression

    // expr(std::shared_ptr<impl> &&shared)
    //     : impl_(std::move(shared)) {
    //     assert(impl_ && "shared pointer cannot be null");
    // }
    /**
     * @brief make a const vector from a pointer
     * @param ptr pointer to the data
     * @return mapped_const_vector
     */
    [[nodiscard]]
    auto make_vec(scalar_t *ptr) const { return mapped_vector(ptr, dim()); }
    /**
     * @brief make a const vector from a pointer
     * @param ptr pointer to the data
     * @return mapped_const_vector
     */
    [[nodiscard]]
    auto make_vec(const scalar_t *ptr) const { return mapped_const_vector(ptr, dim()); }

    /**
     * @brief finalize this expression. Will be called upon added to a problem
     * @warning if one is finalized before dependencies, failure may happen
     * @retval true if successfully finalized
     */
    bool finalize();

    /// @brief clone the expression
    template <typename derived>
        requires std::is_base_of_v<expr, derived>
    derived clone() const {
        derived tmp;
        tmp.impl_.reset(new derived::impl());                                                  // create a new impl
        static_cast<derived::impl &>(*tmp.impl_) = static_cast<const derived::impl &>(*impl_); // copy the shared impl
        if (dep_) {
            tmp.dep_.reset(new expr_list()); // create a new list
            *tmp.dep_ = *dep_;               // copy the dependencies
        }
        return tmp;
    }
    /// @brief moving cast
    /// @tparam derived 
    /// @tparam base 
    /// @return 
    template <typename derived, typename base>
        requires std::is_base_of_v<base, derived>
    derived cast() {
        derived tmp;

        tmp.impl_.reset(new derived::impl(static_cast<base::impl &&>(std::move(*impl_)))); // create a new impl

        tmp.dep_ = std::move(dep_); // move the dependencies
        return tmp;
    }

    template <typename derived>
    derived cast() {
        return cast<derived, expr>();
    } ///< cast to derived type
};
} // namespace moto

#endif /*__EXPRESSION_BASE_*/