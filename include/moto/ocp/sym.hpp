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

    sym *dual_ = nullptr; ///< pointer to the dual sym, e.g., next state in OCP;
    void finalize_impl();

  public:
    using expr::dim;
    using expr::name; ///< name of the symbolic variable
    using expr::shared;

    SHARED_ATTR_GETTER(dual, sym); ///< getter for dual pointer

    sym() = default; ///< default constructor, will create a not-a-number symbolic variable
    /// copy constructor
    sym(const sym &rhs) = default;
    /// move constructor
    sym(sym &&rhs) : cs::SX(std::move(rhs)), expr(std::move(rhs)), dual_(rhs.dual_) {
        if (dual_) {
            rhs.dual_ = nullptr; // clear the dual pointer in the moved object
            dual_->dual() = this; // set the dual pointer
        }
    }
    /**
     * @brief Construct a new sym object
     *
     * @param name name of the symbolic variable
     * @param dim dimension of the symbolic variable
     * @param type type of the symbolic variable, must be one of the symbolic fields
     */
    sym(const std::string &name, size_t dim, field_t type) : expr(name, dim, type) {
        assert(size_t(type) <= field::num_sym || type == __usr_var);
    }
    sym &operator=(sym &&rhs) = default;
    /**
     * @brief check if this sym is equal to another sym
     * only check by pointer to the expression_impl, not by name or uid or cs::SX
     * @param rhs another sym to compare with
     */
    bool operator==(const sym &rhs) { return uid() == rhs.uid(); }
    /// @brief make a symbolic input
    static auto inputs(const std::string &name, size_t dim) {
        return sym(name, dim, __u);
    }
    /// @brief make a symbolic parameter
    static auto params(const std::string &name, size_t dim) {
        return sym(name, dim, __p);
    }
    /// @brief make a pair of symbolic state
    static auto states(const std::string &name, size_t dim) {
        auto temp = sym(name, dim, __x);
        auto next = sym(name + "_nxt", dim, __y);
        temp.dual_ = &next; // set the dual pointer
        next.dual_ = &temp; // set the dual pointer
        return std::make_pair(temp, next);
    }
    static auto state(const std::string &name, size_t dim) {
        auto [x, y] = states(name, dim);
        return x;
    }
    auto &next() const {
        assert(field() == __x && "next() can only be used with __x state to get its dual in __y");
        return *shared()->as<sym>().dual_;
    }
    auto &prev() const {
        assert(field() == __y && "prev() can only be used with __y state to get its dual in __x");
        return *shared()->as<sym>().dual_;
    }
};

using sym_list = expr_list; ///< list of symbolic expressions
} // namespace moto

#endif // MOTO_OCP_SYM_HPP