#ifndef __EXPRESSION_BASE__
#define __EXPRESSION_BASE__

#include <atri/core/fields.hpp>
#include <casadi/casadi.hpp>

namespace atri {
class expr;
def_ptr(expr);
/**
 * @brief general expression base class
 */
class expr {
  private:
    static size_t max_uid; // uid used to index global expressions

  public:
    const std::string name_;
    const size_t dim_;
    const size_t uid_;
    const field_t field_;

    expr(const std::string &name, size_t dim, field_t field)
        : name_(name), dim_(dim), uid_(max_uid++), field_(field) {}

    expr(expr &&rhs)
        : name_(std::move(rhs.name_)), dim_(rhs.dim_), uid_(rhs.uid_), field_(rhs.field_) {}

    auto make_vec(scalar_t *ptr) { return mapped_vector(ptr, dim_); }

    auto make_vec(const scalar_t *ptr) { return mapped_const_vector(ptr, dim_); }

    /**
     * @brief get other variables related to this expression, by default will return empty
     * @return std::vector<expr_ptr_t> list of expressions
     */
    virtual std::vector<expr_ptr_t> get_aux() { return {}; }

    virtual ~expr() = default;
};

namespace cs = casadi;

/**
 * @brief wrapper of symbolic expressions like primal variables or parameters
 * @note in fact a pointer!
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
 * @brief protected vector of expressions
 * @note when get() is called it must not be empty
 */
struct collection {
  private:
    std::vector<expr_ptr_t> expr_;

  protected:
    void add(std::initializer_list<expr_ptr_t> exprs) { expr_.insert(expr_.end(), exprs); }

  public:
    collection(std::initializer_list<expr_ptr_t> exprs) : expr_(exprs) {}
    collection(std::initializer_list<expr *> exprs) {
        for (auto expr : exprs) {
            expr_.emplace_back(expr);
        }
    }
    collection() = default;
    const auto &get() const {
        assert(!expr_.empty());
        return expr_;
    }
};

} // namespace atri

#endif /*__EXPRESSION_BASE_*/