#ifndef MOTO_OCP_SYM_HPP
#define MOTO_OCP_SYM_HPP

#include <moto/core/expr.hpp>
#include <casadi/casadi.hpp>


namespace moto {


namespace cs = casadi;

/**
 * @brief pointer wrapper of symbolic expressions like primal variables or parameters
 * @warning symbolic computation is via cs::SX, so dont cast expr_ptr_t back to sym! (unless you know it is safe)
 */
struct sym : public cs::SX, public impl::shared_<impl::expr, sym> {
    using shared_::operator->; // inherit operator-> to access expr methods
    using shared_::get;
    /**
     * @brief Construct a new sym object
     *
     * @param name name of the symbolic variable
     * @param dim dimension of the symbolic variable
     * @param type type of the symbolic variable, must be one of the symbolic fields
     */
    sym(const std::string &name, size_t dim, field_t type)
        : shared_(new expr_type(name, dim, type)), cs::SX(cs::SX::sym(name, dim)) {
        assert(size_t(type) <= field::num_sym || type == __usr_var);
        if (type == __y) {
            auto &prev_sym = moto::expr_lookup::get<sym>((*this)->uid_ - 1);
            assert(prev_sym->field_ == __x &&
                   prev_sym->name_ + "_nxt" == (*this)->name_ &&
                   "make sure you create a pair of states from sym::states()"); // ensure expr of uid - 1 is the x field
        }
    }
    sym() = default;
    /**
     * @brief check if this sym is equal to another sym
     * only check by pointer to the expression_impl, not by name or uid or cs::SX
     * @param rhs another sym to compare with
     */
    bool operator==(const sym &rhs) {
        return expr_ptr_t::get() == rhs.get();
    }
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
        return std::make_pair(temp, next);
    }
};

} // namespace moto

#endif // MOTO_OCP_SYM_HPP