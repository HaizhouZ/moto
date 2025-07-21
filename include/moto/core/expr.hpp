#ifndef __EXPRESSION_BASE__
#define __EXPRESSION_BASE__

#include <moto/core/fields.hpp>

namespace moto {
namespace impl {
class expr; // forward declaration of expr
} // namespace impl
def_ptr_named(expr, impl::expr);
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
    expr_list(std::initializer_list<impl::expr *> exprs);
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
constexpr size_t dim_tbd = 0;
namespace impl {
/**
 * @brief general expression base class
 */
class expr {
  private:
    static size_t max_uid; /// < uid used to index global expressions
    bool finalized = false;

  protected:
    expr_list dep_;
    virtual void finalize_impl() {}

  public:
    /// @brief abstract base class for expression handle
    struct handle {
        /// @brief get the expression value
        virtual expr &val() const = 0;
        virtual void lock_index_by_name() const = 0;
    };

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
    expr(const std::string &name, size_t dim, field_t field);
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
     * @return pointer to std::vector<expr_ptr_t> list of expressions, default is nullptr
     */
    expr_list &get_dep() { return dep_; }
};
} // namespace impl
/**
 * @brief index of all syms
 *
 */
class expr_lookup {
  public:
    /// all expressions, indexed by uid, nullptr if not finalized (only placeholder)
    using expr_handle = impl::expr::handle;               /// < type of expr handle
    using expr_handle_ptr = std::shared_ptr<expr_handle>; /// < type of expr handle
    static std::vector<expr_handle_ptr> all_;
    /// all expressions, indexed by name
    static std::unordered_map<std::string, expr_handle_ptr> by_name_;
    friend class impl::expr;
    expr_lookup() = delete; // no instance of expr_lookup

  public:
    /// @brief get an expression by name
    template <typename T = expr_handle>
        requires(std::is_base_of_v<expr_handle, T>)
    static T &get(const std::string &name) {
        const auto &p = by_name_[name];
        if (p == nullptr) {
            throw std::runtime_error(fmt::format("expression name {} not locked by shared_handle<T>", name));
        }
        return dynamic_cast<T &>(*p);
    }
    /// @brief get an expression by uid
    template <typename T = expr_handle>
        requires(std::is_base_of_v<expr_handle, T>)
    static T &get(size_t uid) {
        const auto &p = all_[uid];
        if (!p) {
            throw std::runtime_error(fmt::format("expression with uid {} not created from / owned by shared_handle<T>", uid));
        }
        return dynamic_cast<T &>(*p);
    }
};
namespace impl {
/**
 * @brief shared pointer wrapper of expr
 * @tparam derived_impl derived class of expr
 */
template <typename derived, typename derived_shared = void>
    requires std::derived_from<derived, expr>
class shared_handle : public std::shared_ptr<derived>, public expr::handle {
  private:
    /// @brief will copy the derived_shared to expr_lookup::all_[uid], so that the derived_shared class can be accessed by uid
    /// @note this is to enable derived_shared with state, for example, sym with cs::SX::sym
    /// @param uid uid of the expression
    void add_index_by_uid(size_t uid) {
        if constexpr (std::is_void_v<derived_shared>) {
            expr_lookup::all_[uid].reset(new shared_handle(*this));
        } else {
            static_assert(std::is_base_of_v<shared_handle, derived_shared>,
                          "derived_shared must be derived from shared_handle<...>");
            static_assert(std::is_constructible_v<derived_shared, const derived_shared &>,
                          "derived_shared must be copy constructible");
            const auto &cur = static_cast<derived_shared &>(*this);
            expr_lookup::all_[uid].reset(new derived_shared(cur));
        }
        dynamic_cast<shared_handle &>(*expr_lookup::all_[uid]).linked_ = true; // mark this shared_handle as linked to expr_lookup
    }
    /**
     * @brief finalize the expression's global index by name in @ref expr_lookup
     * @param p pointer to the derived object
     * @note will check if the name is already used, if so, will throw an error
     */
    void lock_index_by_name() const override {
        auto p = this->get();
        auto [it, inserted] = expr_lookup::by_name_.try_emplace(p->name_, expr_lookup::all_[p->uid_]);
        if (!inserted) {
            throw std::runtime_error(
                fmt::format("expr name conflicts {} between to-add uid {} and existing uid {}",
                            p->name_, p->uid_, it->second->val().uid_));
        }
    }
    friend class expr;
    bool linked_ = false; // whether this shared_handle is linked to an expr in expr_lookup

  public:
    using expr_type = derived;
    using shared_ptr = std::shared_ptr<derived>;

  protected:
    /// @brief reset the shared pointer to a newly allocated derived object
    /// @param p raw pointer to the derived object from new derived(...)
    void reset(derived *p) {
        // reset the shared pointer
        size_t prev_uid = this->val().uid_;
        shared_ptr::reset(p);
        add_index_by_uid(p->uid_);
    }

  public:
    /// @brief check if this shared_handle is linked to an expr in expr_lookup
    /// @note it helps to distinguish between a normal shared_handle and a linked shared_handle that will be used for expr finalization
    /// @note one is only linked if got from expr_lookup::get()
    bool linked() const { return linked_; }

    expr &val() const override final { return *(*this); }
    /// @brief construct a new shared_handle object pointing to a newly allocated expression
    /// @note will bind a copy of this shared wrapper to the uid of *p globally
    /// @param p raw pointer to the derived object from new derived(...)
    shared_handle(derived *p) : shared_ptr(p) {
        add_index_by_uid((*this)->uid_);
    }
    using shared_ptr::get;
    shared_handle() = default;
};
} // namespace impl
} // namespace moto

#endif /*__EXPRESSION_BASE_*/