#ifndef __EXPRESSION_BASE__
#define __EXPRESSION_BASE__

#include <moto/core/fields.hpp>
#include <casadi/casadi.hpp>

namespace moto {
class expr_impl; // forward declaration of expr_impl
def_ptr_named(expr, expr_impl);
struct expr_list; // forward declaration of expr_list
/**
 * @brief index of all syms
 *
 */
class expr_index {
    /// all expressions, indexed by uid, nullptr if not finalized (only placeholder)
    inline static std::vector<expr_ptr_t> all_;
    /// all expressions, indexed by name
    inline static std::unordered_map<std::string, expr_ptr_t> by_name_{};
    /// allow expr_impl to access private members
    friend class expr_impl;

  public:
    /// @brief get an expression by name
    static const auto &get(const std::string &name) { return by_name_.at(name); }
    /// @brief get an expression by uid
    static const auto &get(size_t uid) { return all_.at(uid); }
};
constexpr size_t dim_tbd = 0;
/**
 * @brief general expression base class
 */
class expr_impl : public std::enable_shared_from_this<expr_impl> {
  private:
    inline static size_t max_uid = 0; /// < uid used to index global expressions
    bool finalized = false;
    void add_to_index();

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
    virtual expr_list *get_aux() { return nullptr; }
};
namespace cs = casadi;

/**
 * @brief pointer wrapper of symbolic expressions like primal variables or parameters
 * @warning symbolic computation is via cs::SX, so dont cast expr_ptr_t back to sym! (unless you know it is safe)
 */
struct sym : public expr_ptr_t, public cs::SX {
    /**
     * @brief Construct a new sym object
     *
     * @param name name of the symbolic variable
     * @param dim dimension of the symbolic variable
     * @param type type of the symbolic variable, must be one of the symbolic fields
     */
    sym(const std::string &name, size_t dim, field_t type)
        : expr_ptr_t(new expr_impl(name, dim, type)), cs::SX(cs::SX::sym(name, dim)) {
        assert(size_t(type) <= field::num_sym || type == __usr_var);
    }
    /**
     * @brief Construct a new sym object from an existing expr_ptr_t
     * @note this will not copy the expr, but use the ones from rhs
     * @param rhs another expr_ptr_t to copy from
     */
    sym(const expr_ptr_t &rhs)
        : expr_ptr_t(rhs), cs::SX(cs::SX::sym(rhs->name_, rhs->dim_)) {}
    sym() = default;
    /**
     * @brief check if this sym is equal to another sym
     * only check by pointer to the expression_impl, not by name or uid or cs::SX
     * @param rhs another sym to compare with
     */
    bool operator==(const sym &rhs) {
        return expr_ptr_t::get() == rhs.get();
    }

    using expr_ptr_t::operator->; // inherit operator-> to access expr_impl methods
    using cs::SX::get;            // deal with the ambiguity of get() in cs::SX and expr_ptr_t
    /**
     * @brief get the underlying expr_impl pointer
     * @return expr_impl pointer
     */
    expr_impl *get() const { return expr_ptr_t::get(); }
};
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
} // namespace moto

#endif /*__EXPRESSION_BASE_*/