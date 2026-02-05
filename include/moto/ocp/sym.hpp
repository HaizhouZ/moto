#ifndef MOTO_OCP_SYM_HPP
#define MOTO_OCP_SYM_HPP

#include <casadi/casadi.hpp>
#include <moto/core/expr.hpp>
#include <moto/utils/func_traits.hpp>

namespace moto {
class sym;
namespace cs = casadi;
/**
 * @brief pointer wrapper of symbolic expressions like primal variables or parameters
 * @note inherits from cs::SX for easy symbolic operations
 */
struct var : public utils::shared<sym>, public cs::SX {
  public:
    using sym = moto::sym;
    using base = utils::shared<sym>;
    using base::get;
    using base::operator->;
    using base::operator bool;
    using base::operator!;
    inline friend bool operator==(const var &lhs, const var &rhs) noexcept;
    var() = default; ///< default constructor
    // forwarding constructors
    template <typename... Args>
        requires std::constructible_from<utils::shared<sym>, Args...>
    var(Args &&...args) : base(std::forward<Args>(args)...), cs::SX((sym &)*this) {}
};
/**
 * @brief pointer wrapper of symbolic expressions like primal variables or parameters
 */
class sym : public expr, public cs::SX {

  public:
    constexpr static char next_suffix_[] = "_nxt"; ///< suffix for next state variable
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

    bool has_non_trivial_integration_ = false; ///< whether the symbolic variable has non-trivial integration
    bool has_non_trivial_difference_ = false;  ///< whether the symbolic variable has non-trivial difference

  public:
    sym() = default;                     ///< default constructor, will create a not-a-number symbolic variable
    sym(sym &&rhs) = default;            ///< move constructor
    sym &operator=(sym &&rhs) = default; ///< move assignment operator

    /**
     * @brief Construct a new sym object
     *
     * @param name name of the symbolic variable
     * @param dim dimension of the symbolic variable
     * @param type type of the symbolic variable, must be one of the symbolic fields
     */
    sym(const std::string &name, size_t dim, field_t type, default_val_t default_val = default_val_none_t());

    using expr::dim;
    using expr::name; ///< name of the symbolic variable
    using expr::operator bool;
    using expr::dep;

    friend bool operator==(const sym &lhs, const sym &rhs) noexcept {
        return operator==((const expr &)lhs, (const expr &)rhs);
    }

    PROPERTY(default_value) ///< default value of the symbolic variable
    PROPERTY(dual)          ///< dual variable, only used for state variables

    CONST_PROPERTY(has_non_trivial_integration) ///< whether the symbolic variable has non-trivial integration
    CONST_PROPERTY(has_non_trivial_difference)  ///< whether the symbolic variable has non-trivial difference

    void set_default_value(const default_val_t &default_val); ///< set the default value of the symbolic variable

    virtual cs::SX symbolic_integrate(const cs::SX &x, const cs::SX &dx) const {
        return x + dx;
    } ///< integrate the variable by dx with step size alpha

    virtual cs::SX symbolic_difference(const cs::SX &x1, const cs::SX &x0) const {
        return x1 - x0;
    } ///< compute the difference between two variables x1 - x0
    /// @brief integrate the variable by dx with step size alpha
    virtual void integrate(vector_ref x, vector_ref dx, vector_ref out, scalar_t alpha = 1.0) const;

    /// @brief compute the difference between two variables x1 - x0
    virtual void difference(vector_ref x1, vector_ref x0, vector_ref out) const;

  protected:
    using integrate_type = utils::func_traits<decltype(&sym::integrate)>::std_func_type;
    std::shared_ptr<integrate_type> integrator_; ///< integrator function
    using difference_type = utils::func_traits<decltype(&sym::difference)>::std_func_type;
    std::shared_ptr<difference_type> differencer_; ///< differencer function

    static void setup_states(const var &x, const var &y) { ///< setup the dual state variables
        x->dual_ = y;
        y->dual_ = x;
        y->dep().clear();
        y->add_dep(x);
    }

  public:
    auto get_next_name() const { return name_ + next_suffix_; } ///< get the name of the next state variable
    /// @brief make a symbolic input
    static var inputs(const std::string &name, size_t dim = 1, default_val_t default_val = default_val_none_t()) {
        return std::make_shared<sym>(name, dim, __u, default_val);
    }
    /// @brief make a symbolic parameter
    static var params(const std::string &name, size_t dim = 1, default_val_t default_val = default_val_none_t()) {
        return std::make_shared<sym>(name, dim, __p, default_val);
    }
    /// @brief make a pair of symbolic state
    static auto states(const std::string &name, size_t dim = 1, default_val_t default_val = default_val_none_t()) {
        auto temp = var(std::make_shared<sym>(name, dim, __x, default_val));
        auto next = var(std::make_shared<sym>(temp->get_next_name(), dim, __y, default_val));
        setup_states(temp, next);
        return std::make_pair(std::move(temp), std::move(next));
    }
    static auto state(const std::string &name, size_t dim = 1, default_val_t default_val = default_val_none_t()) {
        auto [x, y] = states(name, dim, default_val);
        setup_states(x, y);
        return x;
    }
    const var &next() const {
        assert(field_ == __x && "next() can only be used with __x state to get its dual in __y");
        return dual_;
    }
    var &next() {
        assert(field_ == __x && "next() can only be used with __x state to get its dual in __y");
        return dual_;
    }
    const var &prev() const {
        assert(field_ == __y && "dual() can only be used with __y state to get its dual in __x");
        return dual_;
    }
    var &prev() {
        assert(field_ == __y && "dual() can only be used with __y state to get its dual in __x");
        return dual_;
    }

    static var usr_var(const std::string &name, size_t dim = 1, default_val_t default_val = default_val_none_t()) {
        return std::make_shared<sym>(name, dim, __usr_var, default_val);
    } ///< make a user defined variable

    static var symbol(const std::string &name, size_t dim = 1, field_t field = field_t::__undefined, default_val_t default_val = default_val_none_t()) {
        return std::make_shared<sym>(name, dim, field, default_val);
    } ///< make a symbolic primitive

    /// clone the symbolic variable
    virtual var clone(const std::string &name) const;

  protected:
    /// clone the state variable and its dual state
    template <typename derived = sym>
        requires std::is_base_of_v<sym, std::remove_cvref_t<derived>>
    var clone_states(const std::string &name) const {
        var s, d; // new state and its dual
        if (field_ == __x) {
            s = var(new derived((const derived &)*this));
            d = var(new derived((const derived &)dual_));
        } else if (field_ == __y) {
            return dual_->clone_states<derived>(name);
        } else {
            throw std::runtime_error("clone_states can only be used with state variables");
        }
        setup_states(s, d);
        s->name() = name;
        d->name() = s->get_next_name();
        return s;
    }
};

struct var_inarg_list; ///< forward declaration
/// @brief list of symbolic expressions
struct var_list : public std::vector<var> {
    using std::vector<var>::vector; ///< inherit constructors from std::vector
    var_list(const var_inarg_list &v);
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

inline var_list::var_list(const var_inarg_list &v) {
    this->reserve(v.size());
    for (const auto &i : v) {
        this->emplace_back(i.get());
    }
} ///< construct from var_inarg_list
inline bool operator==(const var &lhs, const var &rhs) noexcept {
    return *lhs == *rhs; // use sym's operator==
} ///< operator== for var
} // namespace moto

#endif // MOTO_OCP_SYM_HPP