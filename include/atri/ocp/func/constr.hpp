#ifndef __CONSTR__
#define __CONSTR__

#include <atri/ocp/core/approx.hpp>

namespace atri {
struct constr; // fwd

struct constr_data : public sparse_approx_data {
    vector_ref multiplier_;
    // vector_ref slack_;      /// @todo: add this to raw
    std::vector<row_vector_ref> vjp_;
    constr_data(approx_data *raw, sparse_approx_data &&d, constr *cstr);
};
def_ptr(constr_data);
/**
 * @brief constraint approximation with multipliers (and slack variables)
 */
struct constr : public approx {
    constr(const std::string &name, size_t dim, field_t field,
           approx_order order = approx_order::first,
           bool enable_slack = false)
        : approx(name, dim, field, order) {
        assert(field == __dyn || magic_enum::enum_name(field).find(
                                     "constr") != std::string::npos);
        /// @todo : make dual variables
        // if (enable_slack) {
        // }
    }
    sparse_approx_data_ptr_t make_data(sym_data *primal, approx_data *raw) override {
        return constr_data_ptr_t(
            new constr_data(raw, std::move(*approx::make_data(primal, raw)), this));
    }

  private:
    void value_impl(sparse_approx_data_ptr_t data) override final { value(data); }
    void jacobian_impl(sparse_approx_data_ptr_t data) override final;
    void hessian_impl(sparse_approx_data_ptr_t data) override final;

  protected:
    virtual void value(sparse_approx_data_ptr_t data) {
        approx::value_impl(data);
    }
    virtual void jacobian(sparse_approx_data_ptr_t data) {
        approx::jacobian_impl(data);
    };
    virtual void hessian(constr_data_ptr_t data) {
        approx::hessian_impl(data);
    };
};
def_ptr(constr);
} // namespace atri

#endif