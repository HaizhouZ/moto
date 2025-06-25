#ifndef ATRI_OCP_CORE_SYM_DATA_HPP
#define ATRI_OCP_CORE_SYM_DATA_HPP

#include <atri/ocp/problem.hpp>

namespace atri {
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
            value_[i].resize(prob_->dim_[i]);
            value_[i].setZero();
        }
        for (const auto & v : prob_->expr_[__usr_var]){
            usr_value_[v->uid_] = vector(prob_->dim_[__usr_var]);
            usr_value_[v->uid_].setZero();
        }
    }
    /// get the symbolic variable value of the sym
    vector_ref get(expr_impl *sym) {
        if (sym->field_ == __usr_var)
            return usr_value_[sym->uid_];
        else
            return value_[sym->field_].segment(prob_->get_expr_start(*sym), sym->dim_);
    }
    /// get the symbolic variable value of the sym
    auto get(const sym &sym) {
        return get(sym.get());
    }

    /// pointer to the problem, used to get dimensions of symbolic variables
    ocp_ptr_t prob_;
    /// dense storage of symbolic variables, indexed by field
    std::array<vector, field::num_sym> value_;
    std::unordered_map<size_t, vector> usr_value_;
};

def_unique_ptr(sym_data);
} // namespace atri

#endif // ATRI_OCP_CORE_SYM_DATA_HPP