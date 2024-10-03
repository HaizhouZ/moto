#ifndef __EXPRESSION_BASE__
#define __EXPRESSION_BASE__

#include <atri/core/fields.hpp>

namespace atri {
enum class approx_type {
    zero = 0,
    first,
    second
};

#define def_ptr(name) typedef std::shared_ptr<name> name##_ptr_t;

/**
 * @brief enum class of field types
 *
 */
struct expr {
   private:
    static size_t max_uid;

   public:
    const size_t dim_;
    const std::string& name_;
    const size_t uid_;
    const field_type field_;

    expr(const std::string& name, size_t dim, field_type field)
        : name_(name), dim_(dim), uid_(max_uid++), field_(field) {
    }
    auto from(scalar_t* ptr) { return mapped_vector(ptr, dim_); }

    auto from(const scalar_t* ptr) { return mapped_const_vector(ptr, dim_); }

    virtual ~expr() = default;
};

def_ptr(expr);

struct sym : public expr {
    approx_type type() = delete;
    sym(const std::string& name, size_t dim, field_type type)
        : expr(name, dim, type) {
        assert(size_t(type) <= sym_num);
    }
};

def_ptr(sym);
}  // namespace atri

#endif /*__EXPRESSION_BASE_*/