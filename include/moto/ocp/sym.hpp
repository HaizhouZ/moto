#ifndef MOTO_OCP_SYM_HPP
#define MOTO_OCP_SYM_HPP

#include <casadi/casadi.hpp>
#include <moto/core/expr.hpp>

namespace moto {
namespace cs = casadi;

/**
 * @brief pointer wrapper of symbolic expressions like primal variables or parameters
 */
class sym : public cs::SX, public expr {
    friend class expr; ///< allow expr_lookup to access private members
    sym(expr &&rhs)
        : cs::SX(cs::SX::sym(rhs.name(), rhs.dim())), expr(std::move(rhs)) {} // move constructor from expr
  public:
    using expr::dim;
    using expr::name; ///< name of the symbolic variable
    /**
     * @brief Construct a new sym object
     *
     * @param name name of the symbolic variable
     * @param dim dimension of the symbolic variable
     * @param type type of the symbolic variable, must be one of the symbolic fields
     */
    static sym *create(const std::string &name, size_t dim, field_t type) {
        auto new_ = moving_cast<sym>(expr::create(name, dim, type));
        assert(size_t(type) <= field::num_sym || type == __usr_var);
        if (type == __y) {
            auto *prev_sym = expr_lookup::get<sym>(new_->uid_ - 1);
            assert(prev_sym->field_ == __x &&
                   prev_sym->name_ + "_nxt" == new_->name_ &&
                   "make sure you create a pair of states from sym::states()"); // ensure expr of uid - 1 is the x field
        }
        return new_;
    }
    /**
     * @brief check if this sym is equal to another sym
     * only check by pointer to the expression_impl, not by name or uid or cs::SX
     * @param rhs another sym to compare with
     */
    bool operator==(const sym &rhs) { return uid_ == rhs.uid_; }
    /// @brief make a symbolic input
    static auto inputs(const std::string &name, size_t dim) {
        return create(name, dim, __u);
    }
    /// @brief make a symbolic parameter
    static auto params(const std::string &name, size_t dim) {
        return create(name, dim, __p);
    }
    /// @brief make a pair of symbolic state
    static auto states(const std::string &name, size_t dim) {
        auto temp = create(name, dim, __x);
        auto next = create(name + "_nxt", dim, __y);
        return std::make_pair(temp, next);
    }
    static auto state(const std::string &name, size_t dim) {
        auto [x, y] = states(name, dim);
        return x;
    }
    auto *next() const {
        assert(field_ == __x && "next() can only be used with __x state to get its dual in __y");
        return expr_lookup::get<sym>(uid_ + 1);
    }
    auto *prev() const {
        assert(field_ == __y && "prev() can only be used with __y state to get its dual in __x");
        return expr_lookup::get<sym>(uid_ - 1);
    }
};
def_raw_ptr(sym);
using sym_list = std::vector<sym*>;
using sym_init_list = std::initializer_list<sym_list::value_type>;
} // namespace moto

#endif // MOTO_OCP_SYM_HPP