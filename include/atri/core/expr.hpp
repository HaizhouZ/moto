#ifndef __EXPRESSION_BASE__
#define __EXPRESSION_BASE__

#include <atri/core/fields.hpp>

namespace atri {
enum class approx_order {
    zero = 0,
    first,
    second
};

#define def_ptr(name) typedef std::shared_ptr<name> name##_ptr_t;

/**
 * @brief enum class of field types
 *
 */
class expr {
   private:
    static size_t max_uid;

   public:
    const size_t dim_;
    const std::string& name_;
    const size_t uid_;
    const field::type field_;

    expr(const std::string& name, size_t dim, field::type field)
        : name_(name), dim_(dim), uid_(max_uid++), field_(field) {
    }
    auto make_vec(scalar_t* ptr) { return mapped_vector(ptr, dim_); }

    auto make_vec(const scalar_t* ptr) { return mapped_const_vector(ptr, dim_); }

    virtual ~expr() = default;
};

def_ptr(expr);

struct sym : public expr {
    approx_order type() = delete;
    sym(const std::string& name, size_t dim, field::type type)
        : expr(name, dim, type) {
        assert(size_t(type) <= field::num_sym);
    }
};

def_ptr(sym);
}  // namespace atri

#endif /*__EXPRESSION_BASE_*/