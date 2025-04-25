#ifndef __CONSTR__
#define __CONSTR__

#include <atri/ocp/approx.hpp>

namespace atri {
struct constr_data : public approx_data {
    vector multiplier_;
    vector slack_;
    constr_data(approx_data &&d) : approx_data(std::move(d)) {
        multiplier_.resize(d.v_.size());
        slack_.resize(d.v_.size());
    }
};
def_ptr(constr_data);
/**
 * @brief constraint approximation with multipliers (and slack variables)
 */
struct constr : public approx {
    constr(const std::string &name, size_t dim, field::type field,
           approx_order order = approx_order::first)
        : approx(name, dim, field, order) {
        assert(field == field::dyn || field == field::eq_constr ||
               field == field::ineq_constr);
    };
    approx_data_ptr_t make_data(primal_data &raw) override {
        return constr_data_ptr_t(
            new constr_data(std::move(*approx::make_data(raw))));
    };
};

} // namespace atri

#endif