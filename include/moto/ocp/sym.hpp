#ifndef MOTO_OCP_SYM_HPP
#define MOTO_OCP_SYM_HPP

#include <casadi/casadi.hpp>
#include <moto/core/expr.hpp>

namespace moto {
class sym;
namespace cs = casadi;
struct var : public shared_expr, public cs::SX {
  public:
    using base = shared_expr;
    using sym = moto::sym;
    sym *operator->() const; ///< convert to sym
    var() = default;         ///< default constructor, will create a not-a-number symbolic variable
    template <typename T, typename T_ = std::remove_cvref_t<T>>
    var(T &&rhs) noexcept : base(std::forward<T>(rhs)), cs::SX((sym &)*this) {}
    var(var &&rhs) noexcept : base(std::move(rhs)), cs::SX(static_cast<cs::SX &&>(rhs)) {}
    var(const var &) = default;                ///< copy constructor
    var &operator=(const var &) = default;     ///< copy assignment operator
    var &operator=(var &&) noexcept = default; ///< move assignment operator
};
/**
 * @brief pointer wrapper of symbolic expressions like primal variables or parameters
 */
class sym : public expr, public cs::SX {

  public:
    friend class expr;
    using default_val_t = std::variant<vector, scalar_t, std::monostate>;
    using default_val_none_t = std::monostate;

  protected:
    vector default_value_;
    var dual_;

    void finalize_impl() override;
    operator double() const = delete;     ///< disable implicit conversion to double
    operator casadi_int() const = delete; ///< disable implicit conversion to casadi_int

    sym(const sym &rhs) = default;            ///< copy constructor
    sym &operator=(const sym &rhs) = default; ///< copy assignment operator

    friend class shared_expr; ///< allow shared_expr to access private members
    friend class var;         ///< allow var to access private members

    sym() = default; ///< default constructor, will create a not-a-number symbolic variable
    /**
     * @brief Construct a new sym object
     *
     * @param name name of the symbolic variable
     * @param dim dimension of the symbolic variable
     * @param type type of the symbolic variable, must be one of the symbolic fields
     */
    sym(const std::string &name, size_t dim, field_t type,
        default_val_t default_val = default_val_none_t())
        : expr(name, dim, type), cs::SX(cs::SX::sym(name, dim)) {
        assert(size_t(type) <= field::num_sym || type == __usr_var);
        if (std::holds_alternative<vector>(default_val)) {
            default_value_ = std::move(std::get<vector>(default_val));
        } else if (std::holds_alternative<scalar_t>(default_val)) {
            default_value_ = vector::Constant(dim, std::get<scalar_t>(default_val));
        } /// leave empty
    }

    sym(sym &&rhs) = default;            ///< move constructor
    sym &operator=(sym &&rhs) = default; ///< move assignment operator

  public:
    using expr::dim;
    using expr::name; ///< name of the symbolic variable
    using expr::operator bool;

    PROPERTY(default_value) ///< default value of the symbolic variable

    /// @brief make a symbolic input
    static var inputs(const std::string &name, size_t dim, default_val_t default_val = default_val_none_t()) {
        return sym(name, dim, __u, default_val);
    }
    /// @brief make a symbolic parameter
    static var params(const std::string &name, size_t dim, default_val_t default_val = default_val_none_t()) {
        return sym(name, dim, __p, default_val);
    }
    /// @brief make a pair of symbolic state
    static auto states(const std::string &name, size_t dim, default_val_t default_val = default_val_none_t()) {
        auto temp = var(sym(name, dim, __x, default_val));
        auto next = var(sym(name + "_nxt", dim, __y, default_val));
        temp->dual_ = next;
        next->dual_ = temp;
        return std::make_pair(std::move(temp), std::move(next));
    }
    static auto state(const std::string &name, size_t dim, default_val_t default_val = default_val_none_t()) {
        auto [x, y] = states(name, dim, default_val);
        return x;
    }
    var &next() {
        assert(field_ == __x && "next() can only be used with __x state to get its dual in __y");
        return dual_;
    }
    var &prev() {
        assert(field_ == __y && "dual() can only be used with __y state to get its dual in __x");
        return dual_;
    }

    static var usr_var(const std::string &name, size_t dim, default_val_t default_val = default_val_none_t()) {
        return sym(name, dim, __usr_var, default_val);
    } ///< make a user defined variable

    static var symbol(const std::string &name, size_t dim, field_t field, default_val_t default_val = default_val_none_t()) {
        return sym(name, dim, field, default_val);
    } ///< make a symbolic primitive
};
inline sym *var::operator->() const {
    return static_cast<sym *>(base::operator->());
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