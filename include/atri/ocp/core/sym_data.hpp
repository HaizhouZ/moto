#ifndef ATRI_OCP_CORE_SYM_DATA_HPP
#define ATRI_OCP_CORE_SYM_DATA_HPP

#include <atri/ocp/core/problem.hpp>

namespace atri {

struct sym_data {
    sym_data(problem_ptr_t prob) : prob_(prob) {
        for (size_t i = 0; i < field::num_sym; i++) {
            value_[i].resize(prob_->dim_[i]);
        }
    }

    auto get(const sym& sym) {
        return sym->make_vec(
            prob_->get_data_ptr(value_[sym->field_].data(), sym));
    }

    problem_ptr_t prob_;
    std::array<vector, field::num_sym> value_;
};
} // namespace atri

#endif // ATRI_OCP_CORE_SYM_DATA_HPP