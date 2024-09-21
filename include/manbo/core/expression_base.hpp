#ifndef __EXPRESSION_BASE__
#define __EXPRESSION_BASE__

#include <manbo/common/fwd.hpp>
#include <manbo/core/enums.hpp>
#include <memory>

namespace manbo {
enum class approx_type {
    zero = 0,
    first,
    second
};
/**
 * @brief enum class of field types
 *
 */

struct expr_base {
   private:
    static size_t max_uid;

   public:
    const size_t dim_;
    const std::string& name_;
    const size_t uid_;
    const field_type field_;

    expr_base(const std::string& name, size_t dim, field_type field)
        : name_(name), dim_(dim), uid_(max_uid++), field_(field) {
    }
    auto from(scalar_t* ptr) { return mapped_vector(ptr, dim_); }

    auto from(const scalar_t* ptr) { return mapped_const_vector(ptr, dim_); }

    virtual ~expr_base() = default;
};
typedef std::shared_ptr<expr_base> expr_ptr_t;

struct symbolic : public expr_base {
    static constexpr size_t sym_num = size_t(field_type::p);
    approx_type type() = delete;
    symbolic(const std::string& name, size_t dim, field_type type)
        : expr_base(name, dim, type) {
        assert(size_t(type) <= sym_num);
    }
};

typedef std::shared_ptr<symbolic> sym_ptr_t;
}  // namespace manbo

#endif /*__EXPRESSION_BASE_*/