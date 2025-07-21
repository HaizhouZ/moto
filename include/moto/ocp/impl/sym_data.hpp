#ifndef MOTO_OCP_CORE_SYM_DATA_HPP
#define MOTO_OCP_CORE_SYM_DATA_HPP

#include <moto/ocp/problem.hpp>
#include <moto/ocp/sym.hpp>

namespace moto {
/**
 * @brief Symbolic data storage
 * stores the symbolic variables in a dense format
 * and provides access to the values of the symbolic variables
 *
 */
struct sym_data {
    /**
     * @brief Construct a new sym data object
     *
     * @param prob problem pointer, it will be used to get the dimensions of the symbolic variables
     */
    sym_data(const ocp_ptr_t &prob) : prob_(prob) {
        for (size_t i = 0; i < field::num_sym; i++) {
            value_[i].resize(prob_->dim(i));
            value_[i].setZero();
        }
        for (const auto &v : prob_->exprs(__usr_var)) {
            usr_value_[v->uid()] = vector(prob_->dim(__usr_var));
            usr_value_[v->uid()].setZero();
        }
    }
    /// get the symbolic variable value of the sym
    vector_ref get(const sym *sym) {
        if (sym->field() == __usr_var)
            return usr_value_.at(sym->uid());
        else
            return prob_->extract(value_.at(sym->field()), sym);
    }

    auto operator[](const sym *sym) {
        return get(sym);
    }

    /// pointer to the problem, used to get dimensions of symbolic variables
    ocp_ptr_t prob_;
    /// dense storage of symbolic variables, indexed by field
    std::array<vector, field::num_sym> value_;
    std::unordered_map<size_t, vector> usr_value_;
};

def_unique_ptr(sym_data);
} // namespace moto

#endif // MOTO_OCP_CORE_SYM_DATA_HPP