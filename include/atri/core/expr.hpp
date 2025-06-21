#ifndef __EXPRESSION_BASE__
#define __EXPRESSION_BASE__

#include <atri/core/fields.hpp>
#include <casadi/casadi.hpp>

namespace atri {
class expr;
def_ptr(expr);
struct expr_list; // forward declaration of expr_list
/**
 * @brief index of all syms
 *
 */
class expr_index {
    inline static std::vector<expr_ptr_t> all_;
    inline static std::unordered_map<std::string, expr_ptr_t> by_name_{};
    friend class expr;

  public:
    static const auto &get(const std::string &name) { return by_name_.at(name); }
    static const auto &get(size_t uid) { return all_.at(uid); }
};
/**
 * @brief general expression base class
 */
class expr : public std::enable_shared_from_this<expr> {
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
    expr(const std::string &name, size_t dim, field_t field)
        : name_(name), dim_(dim), uid_(max_uid++), field_(field) { expr_index::all_.push_back(nullptr); }

    expr(const expr &rhs) = delete;
    expr(expr &&rhs)
        : name_(std::move(rhs.name_)), dim_(rhs.dim_), uid_(rhs.uid_), field_(rhs.field_) { expr_index::all_.push_back(nullptr); }

    auto make_vec(scalar_t *ptr) { return mapped_vector(ptr, dim_); }

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
    virtual expr_list* get_aux() { return nullptr; }
};

namespace cs = casadi;

/**
 * @brief pointer wrapper of symbolic expressions like primal variables or parameters
 * @warning symbolic computation is via cs::SX, so dont cast expr_ptr_t back to sym! (unless you know it is safe)
 */
struct sym : public expr_ptr_t, public cs::SX {
    sym(const std::string &name, size_t dim, field_t type)
        : expr_ptr_t(new expr(name, dim, type)), cs::SX(cs::SX::sym(name, dim)) {
        assert(size_t(type) <= field::num_sym);
    }
    sym(const expr_ptr_t &rhs)
        : expr_ptr_t(rhs), cs::SX(cs::SX::sym(rhs->name_, rhs->dim_)) {}
    sym() = default;

    bool operator==(const sym &rhs) {
        return this->ptr() == rhs.ptr();
    }

    bool empty() { return this->ptr() == nullptr; }

    using expr_ptr_t::operator->;
    expr *ptr() const { return expr_ptr_t::get(); }
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
    expr_list(std::initializer_list<expr *> exprs) {
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
} // namespace atri

#endif /*__EXPRESSION_BASE_*/