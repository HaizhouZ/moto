#ifndef MOTO_OCP_SYM_HPP
#define MOTO_OCP_SYM_HPP

#include <casadi/casadi.hpp>
#include <moto/core/expr.hpp>

namespace moto {
namespace cs = casadi;

/**
 * @brief pointer wrapper of symbolic expressions like primal variables or parameters
 */
class sym : public expr, public cs::SX {

  protected:
    friend class expr; ///< allow expr_lookup to access private members
    shared_expr dual_; ///< pointer to the dual sym, e.g., next state in OCP;

    struct impl : public expr::impl, public cs::SX {
        std::weak_ptr<impl> dual_; ///< pointer to the dual sym, e.g., next state in OCP;
        impl(expr::impl &&rhs)
            : expr::impl(std::move(rhs)), cs::SX(cs::SX::sym(name_, dim_)) {} ///< move constructor
        impl(impl &&rhs)
            : expr::impl(std::move(rhs)), cs::SX(std::move(rhs)), dual_(rhs.dual_) {} ///< move constructor)
    };

    void finalize_impl() override;
    DEF_IMPL_GETTER();

  public:
    using expr::dim;
    using expr::name; ///< name of the symbolic variable

    sym() = default; ///< default constructor, will create a not-a-number symbolic variable
    /**
     * @brief Construct a new sym object
     *
     * @param name name of the symbolic variable
     * @param dim dimension of the symbolic variable
     * @param type type of the symbolic variable, must be one of the symbolic fields
     */
    sym(const std::string &name, size_t dim, field_t type) : expr(name, dim, type) {
        assert(size_t(type) <= field::num_sym || type == __usr_var);
        impl_.reset(new impl(std::move(*impl_)));
        static_cast<cs::SX &>(*this) = get_impl();
    }
    /// @brief Construct a new sym object from an existing expr
    /// @note it is assumed that the expr pointing to a @ref sym::impl
    template <typename T>
        requires std::is_same_v<expr, std::remove_cvref_t<T>>
    sym(T &&rhs) : expr(std::forward<T>(rhs)) {
        impl_.reset(new impl(std::move(*impl_)));
        static_cast<cs::SX &>(*this) = get_impl();
    } ///< move constructor from expr

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
        temp.dual_ = next; // set the dual pointer
        next.get_impl().dual_ = std::static_pointer_cast<impl>(temp.impl_);
        return std::make_pair(temp, next);
    }
    static auto state(const std::string &name, size_t dim) {
        auto [x, y] = states(name, dim);
        return x;
    }
    sym &next() const {
        assert(field() == __x && "next() can only be used with __x state to get its dual in __y");
        return dual_; // get the shared pointer of the dual
    }
    sym prev() const { /// restrictive implementation, only for __y state
        assert(field() == __y && "dual() can only be used with __y state to get its dual in __x");
        sym tmp;
        tmp.impl_ = std::move(get_impl().dual_.lock()); // create a new impl
        return tmp;                                     // get the shared pointer of the dual
    }
};

using sym_list = std::vector<sym>; ///< list of symbolic expressions
} // namespace moto

#endif // MOTO_OCP_SYM_HPP