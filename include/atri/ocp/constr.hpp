#ifndef __CONSTR__
#define __CONSTR__

#include <atri/ocp/approx.hpp>

namespace atri {
struct constr_data : public sparse_approx_data {
    vector multiplier_;
    vector slack_;
    constr_data(sparse_approx_data &&d) : sparse_approx_data(std::move(d)) {
        multiplier_.resize(d.v_.size());
        slack_.resize(d.v_.size());
    }
};
def_ptr(constr_data);
/**
 * @brief constraint approximation with multipliers (and slack variables)
 */
struct constr : public approx {
    constr(const std::string &name, size_t dim, field_t field,
           approx_order order = approx_order::first)
        : approx(name, dim, field, order) {
        assert(field == __dyn || magic_enum::enum_name(field).find(
                                          "constr") != std::string::npos);
    }
    sparse_approx_data_ptr_t make_data(raw_data &raw) override {
        return constr_data_ptr_t(
            new constr_data(std::move(*approx::make_data(raw))));
    }
};

} // namespace atri

#endif