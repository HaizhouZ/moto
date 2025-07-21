#ifndef __EXPRESSION_BASE__
#define __EXPRESSION_BASE__

#include <moto/core/fields.hpp>

namespace moto {
class expr; // forward declaration of expr
def_raw_ptr(expr);
using expr_list = std::vector<expr *>;

constexpr size_t dim_tbd = 0;

#define CONST_ATTR_GETTER(mem_name) \
    inline const auto &mem_name() const { return mem_name##_; }

template <typename labeled_type>
struct uid_t {
    static size_t max_uid;                    ///< uid used to index global expressions
    size_t value = 0;                         ///< unique id of the expression, used to index in expr_lookup
    operator size_t() const { return value; } ///< allow implicit conversion to size_t
    uid_t() : value(max_uid++) {}             ///< default constructor, increment max_uid
    uid_t(const uid_t &rhs) = delete;         ///< delete copy constructor
    uid_t(uid_t &&rhs) : value(rhs.value) { rhs.value = std::numeric_limits<size_t>::max(); }
};

#define INIT_UID_(type) template <> \
                        size_t uid_t<type>::max_uid = 0;

template <typename T>
inline auto format_as(const uid_t<T> &uid) { return size_t(uid); }
/**
 * @brief general expression base class
 */
class expr {
  public:
    /// @brief abstract base class for expression handle
    using handle_t = std::unique_ptr<expr>;

  private:
    static size_t max_uid; /// < uid used to index global expressions
    bool finalized = false;

  protected:
    std::string name_; ///< name of the expression
    size_t dim_;       ///< dimension of the expression, 0 if not set
    uid_t<expr> uid_;  ///< unique id of the expression, used to index in expr_lookup
    field_t field_;    ///< field of the expression, default is field_t::__undefined (i.e., undecided)
    expr_list dep_;    ///< dependencies of the expression, i.e., other expressions that this expression depends on
    virtual void finalize_impl() {}
    expr() = default; // default constructor for derived classes
    expr(expr &&rhs)
        : name_(std::move(rhs.name_)), dim_(rhs.dim_), uid_(std::move(rhs.uid_)),
          dep_(std::move(rhs.dep_)), field_(rhs.field_) {
    }

    expr(const expr &rhs) = delete;

    handle_t *hd_ = nullptr;

    void update_name(const std::string &name);

  public:
    CONST_ATTR_GETTER(name);  ///< getter for name
    CONST_ATTR_GETTER(dim);   ///< getter for dim
    CONST_ATTR_GETTER(uid);   ///< getter for uid
    CONST_ATTR_GETTER(field); ///< getter for field

    /**
     * @brief Construct a new expr
     * @note by default set field to field_t::NUM (i.e., undecided), also by default dim = 0
     * @param name name of the expression
     * @param dim dimension
     * @param field
     */
    static expr *create(const std::string &name, size_t dim, field_t field);

    template <typename derived, typename rhs_derived>
        requires(std::is_base_of_v<expr, derived> && std::is_base_of_v<expr, rhs_derived>)
    static derived *moving_cast(rhs_derived *rhs) {
        auto new_ = new derived(std::move(*rhs));
        new_->hd_ = rhs->hd_;   // transfer ownership of handle
        new_->hd_->reset(new_); // reset the handle to point to the new object. Note this will delete the old object
        return new_;
    }
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
     * @warning if one is finalized before dependencies, failure may happen
     * @retval true if successfully finalized
     */
    bool finalize();
    /**
     * @brief get other variables related to this expression
     */
    CONST_ATTR_GETTER(dep); ///< getter for dep
};
/**
 * @brief index of all syms
 *
 */
class expr_lookup {
  public:
    /// all expressions, indexed by uid, nullptr if not finalized (only placeholder)
    using expr_handle = expr::handle_t; /// < type of expr handle
    static std::vector<expr_handle> all_;
    /// all expressions, indexed by name
    static std::unordered_map<std::string, expr_handle *> by_name_;
    friend class expr;
    expr_lookup() = delete; // no instance of expr_lookup

  public:
    /// @brief get an expression by name
    template <typename T = expr>
        requires(std::is_base_of_v<expr, T>)
    static T *get(const std::string &name) {
        const auto &p = by_name_[name];
        if (p == nullptr) {
            throw std::runtime_error(fmt::format("expression name {} does not exist", name));
        }
        return dynamic_cast<T *>(*p);
    }
    /// @brief get an expression by uid
    template <typename T = expr>
        requires(std::is_base_of_v<expr, T>)
    static T *get(size_t uid) {
        const auto &p = all_[uid];
        if (!p) {
            throw std::runtime_error(fmt::format("expression with uid {} not created from / owned by shared_handle<T>", uid));
        }
        return dynamic_cast<T *>(p.get());
    }
};

} // namespace moto

#endif /*__EXPRESSION_BASE_*/