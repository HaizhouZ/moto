#ifndef MOTO_OCP_SYM_HPP
#define MOTO_OCP_SYM_HPP

#include <casadi/casadi.hpp>
#include <moto/core/expr.hpp>

namespace moto {
namespace cs = casadi;
class sym;
struct var : public shared_expr {
    using base = shared_expr;
    using base::base;
    sym *operator->() const;         ///< convert to sym
    operator sym &() const noexcept; ///< convert to sym
};
/**
 * @brief pointer wrapper of symbolic expressions like primal variables or parameters
 */
class sym : public expr, public cs::SX {

  public:
    friend class expr;

  protected:
    var dual_;
    void finalize_impl() override;
    operator double() const = delete; ///< disable implicit conversion to double
    operator casadi_int() const = delete; ///< disable implicit conversion to casadi_int

  public:
    using expr::dim;
    using expr::name; ///< name of the symbolic variable
    using expr::operator bool;

    sym() = default; ///< default constructor, will create a not-a-number symbolic variable
    /**
     * @brief Construct a new sym object
     *
     * @param name name of the symbolic variable
     * @param dim dimension of the symbolic variable
     * @param type type of the symbolic variable, must be one of the symbolic fields
     */
    sym(const std::string &name, size_t dim, field_t type)
        : expr(name, dim, type), cs::SX(cs::SX::sym(name, dim)) {
        assert(size_t(type) <= field::num_sym || type == __usr_var);
    }
    /// @brief Construct a new sym object from an existing expr
    /// @note it is assumed that the expr pointing to a @ref sym::impl
    template <typename T>
        requires std::is_same_v<expr, std::remove_cvref_t<T>>
    sym(T &&rhs) : expr(std::forward<T>(rhs)) {}

    sym(const sym &rhs) = default;            ///< copy constructor
    sym(sym &&rhs) = default;                 ///< move constructor
    sym &operator=(const sym &rhs) = default; ///< copy assignment operator
    sym &operator=(sym &&rhs) = default;      ///< move assignment operator

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
        auto temp = var(sym(name, dim, __x));
        auto next = var(sym(name + "_nxt", dim, __y));
        temp->dual_ = next;
        next->dual_ = temp;
        return std::make_pair(std::move(temp), std::move(next));
    }
    static auto state(const std::string &name, size_t dim) {
        auto [x, y] = states(name, dim);
        return x;
    }
    sym &next() const {
        assert(field_ == __x && "next() can only be used with __x state to get its dual in __y");
        return dual_;
    }
    sym &prev() const {
        assert(field_ == __y && "dual() can only be used with __y state to get its dual in __x");
        return dual_;
    }
};
inline sym *var::operator->() const {
    return static_cast<sym *>(base::operator->());
}
inline var::operator sym &() const noexcept {
    return *static_cast<sym *>(base::operator->());
}
struct var_list : public std::vector<var> {
    using std::vector<var>::vector; ///< inherit constructors from std::vector
}; ///< list of symbolic expressions
struct var_inarg_list : public std::vector<std::reference_wrapper<sym>> {
    using std::vector<std::reference_wrapper<sym>>::vector; ///< inherit constructors from std::vector
    var_inarg_list(const var_list &v) {
        this->reserve(v.size());
        for (const auto &i : v) {
            this->emplace_back(i);
        }
    } ///< construct from var_list
}; ///< list of symbolic expressions
} // namespace moto

#endif // MOTO_OCP_SYM_HPP