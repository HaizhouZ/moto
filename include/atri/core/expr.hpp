#ifndef __EXPRESSION_BASE__
#define __EXPRESSION_BASE__

#include <atri/core/fields.hpp>

namespace atri {
enum class approx_order { zero = 0,
                          first,
                          second };

/**
 * @brief enum class of field types
 *
 */
class expr {
  private:
    static size_t max_uid;

  public:
    const size_t dim_;
    const std::string name_;
    const size_t uid_;
    const field_t field_;

    expr(const std::string &name, size_t dim, field_t field)
        : name_(name), dim_(dim), uid_(max_uid++), field_(field) {}
    auto make_vec(scalar_t *ptr) { return mapped_vector(ptr, dim_); }

    auto make_vec(const scalar_t *ptr) { return mapped_const_vector(ptr, dim_); }

    virtual ~expr() = default;
};

def_ptr(expr);

struct sym : public expr {
    sym(const std::string &name, size_t dim, field_t type)
        : expr(name, dim, type) {
        assert(size_t(type) <= field::num_sym);
    }
};
def_ptr(sym);

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