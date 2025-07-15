#ifndef __EXPRESSION_BASE__
#define __EXPRESSION_BASE__

#include <casadi/casadi.hpp>
#include <moto/core/fields.hpp>

namespace moto {
namespace impl {
class expr; // forward declaration of expr
/**
 * @brief abstract base class for expression handle
 *
 */
struct expr_handle {
    /// @brief get the expression value
    virtual expr &val() const = 0;
    /**
     * @brief add expression to the expr_lookup
     * @param p pointer to the derived object
     * @note will check if the name is already used, if so, should throw an error
     */
    virtual void finalize_index_by_name(const expr *p) const = 0;
};
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
/**
 * @brief index of all syms
 *
 */
class expr_lookup {
  protected:
    using expr_handle = impl::expr_handle;
    using shared_expr_handle_ptr = std::unique_ptr<expr_handle>;
    /// all expressions, indexed by uid, nullptr if not finalized (only placeholder)
    inline static std::vector<shared_expr_handle_ptr> all_;
    /// all expressions, indexed by name
    inline static std::unordered_map<std::string, expr_handle &> by_name_{};
    friend class impl::expr;

  public:
    /// @brief get an expression by name
    template <typename T = expr_handle>
        requires(std::is_base_of_v<expr_handle, T>)
    static const T &get(const std::string &name) {
        try {
            auto &p = by_name_.at(name);
            return dynamic_cast<T &>(p);
        } catch (const std::out_of_range &e) {
            throw std::runtime_error(fmt::format("expression with name {} not found", name));
        }
    }
    /// @brief get an expression by uid
    template <typename T = expr_handle>
        requires(std::is_base_of_v<expr_handle, T>)
    static const auto &get(size_t uid) {
        auto &p = all_[uid];
        if (all_.at(uid) == nullptr) {
            throw std::runtime_error(fmt::format("expression with uid {} not created from / owned by shared_<T>", uid));
        }
        return dynamic_cast<T &>(*p);
    }
};
constexpr size_t dim_tbd = 0;
namespace impl {
/**
 * @brief general expression base class
 */
class expr {
  private:
    inline static size_t max_uid = 0; /// < uid used to index global expressions
    bool finalized = false;

  protected:
    expr_list dep_;
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
    expr(const std::string &name, size_t dim, field_t field)
        : name_(name), dim_(dim), uid_(max_uid++), field_(field) { expr_lookup::all_.push_back(nullptr); }
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
    expr_list &get_dep() { return dep_; }
};
/**
 * @brief shared pointer wrapper of expr
 * @tparam derived_impl derived class of expr
 */
template <typename derived, typename derived_shared = void>
    requires std::derived_from<derived, expr>
class shared_ : public std::shared_ptr<derived>, public expr_handle, private expr_lookup {
  private:
    /// @brief will copy the derived_shared to expr_lookup::all_[uid], so that the derived_shared class can be accessed by uid
    /// @note this is to enable derived_shared with state, for example, sym with cs::SX::sym
    /// @param uid uid of the expression
    void add_index_by_uid(size_t uid) {
        if constexpr (std::is_void_v<derived_shared>) {
            expr_lookup::all_[uid].reset(new shared_(*this));
        } else {
            static_assert(std::is_base_of_v<shared_, derived_shared>,
                          "derived_shared must be derived from shared_<...>");
            static_assert(std::is_constructible_v<derived_shared, const derived_shared &>,
                          "derived_shared must be copy constructible");
            const auto &cur = static_cast<derived_shared &>(*this);
            expr_lookup::all_[uid].reset(new derived_shared(cur));
        }
        dynamic_cast<shared_&>(*expr_lookup::all_[uid]).linked_ = true; // mark this shared_ as linked to expr_lookup
    }
    /**
     * @brief finalize the expression's global index by name in @ref expr_lookup
     * @param p pointer to the derived object
     * @note will check if the name is already used, if so, will throw an error
     */
    void finalize_index_by_name(const expr *p) const {
        if (!linked()) /// only the linked shared_ can be used to finalize the index
            throw std::runtime_error(fmt::format("shared_ of {} is not linked to expr_lookup", p->name_));
        auto [it, inserted] = expr_lookup::by_name_.try_emplace(p->name_, *expr_lookup::all_[p->uid_]);
        if (!inserted and it->second.val().uid_ != p->uid_) {
            throw std::runtime_error(
                fmt::format("expr name conflicts {} of uid {} with existing uid {}",
                            p->name_, p->uid_, it->second.val().uid_));
        }
    }
    friend class expr;

    bool linked_ = false; // whether this shared_ is linked to an expr in expr_lookup

  public:
    using expr_type = derived;
    using shared_ptr = std::shared_ptr<derived>;

  protected:
    /// @brief reset the shared pointer to a newly allocated derived object
    /// @param p raw pointer to the derived object from new derived(...)
    void reset(derived *p) {
        // reset the shared pointer
        shared_ptr::reset(p);
        add_index_by_uid(p->uid_);
    }

  public:
    /// @brief check if this shared_ is linked to an expr in expr_lookup
    /// @note it helps to distinguish between a normal shared_ and a linked shared_ that will be used for expr finalization
    /// @note one is only linked if got from expr_lookup::get()
    bool linked() const { return linked_; }

    expr &val() const override final { return *(*this); }
    /// @brief construct a new shared_ object pointing to a newly allocated expression
    /// @note will bind a copy of this shared wrapper to the uid of *p globally
    /// @param p raw pointer to the derived object from new derived(...)
    shared_(derived *p) : shared_ptr(p) {
        add_index_by_uid((*this)->uid_);
    }
    using shared_ptr::get;
    shared_() = default;
};
} // namespace impl

inline expr_list::expr_list(std::initializer_list<impl::expr *> exprs) {
    for (auto expr : exprs) {
        emplace_back(impl::shared_<impl::expr>(expr));
    }
}

namespace cs = casadi;

/**
 * @brief pointer wrapper of symbolic expressions like primal variables or parameters
 * @warning symbolic computation is via cs::SX, so dont cast expr_ptr_t back to sym! (unless you know it is safe)
 */
struct sym : public cs::SX, public impl::shared_<impl::expr, sym> {
    using shared_::operator->; // inherit operator-> to access expr methods
    using shared_::get;
    /**
     * @brief Construct a new sym object
     *
     * @param name name of the symbolic variable
     * @param dim dimension of the symbolic variable
     * @param type type of the symbolic variable, must be one of the symbolic fields
     */
    sym(const std::string &name, size_t dim, field_t type)
        : shared_(new expr_type(name, dim, type)), cs::SX(cs::SX::sym(name, dim)) {
        assert(size_t(type) <= field::num_sym || type == __usr_var);
        if (type == __y) {
            auto &prev_sym = moto::expr_lookup::get<sym>((*this)->uid_ - 1);
            assert(prev_sym->field_ == __x &&
                   prev_sym->name_ + "_nxt" == (*this)->name_ &&
                   "make sure you create a pair of states from make_state()"); // ensure expr of uid - 1 is the x field
        }
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
};
} // namespace moto

#endif /*__EXPRESSION_BASE_*/