#ifndef ATRI_CONSTR_HPP
#define ATRI_CONSTR_HPP

#include <atri/ocp/approx.hpp>

namespace atri {
struct constr_impl; // fwd
/**
 * @brief constraint data
 * derived from sparse_approx_data with multipler and vjp (for cost) mapping in addition
 */
struct constr_data : public sparse_approx_data {
    /// @todo: add this to raw
    // vector_ref slack_;      
    vector_ref multiplier_;
    std::vector<row_vector_ref> vjp_;
    constr_data(approx_storage *raw, sparse_approx_data &&d, constr_impl *cstr);
};
def_unique_ptr(constr_data);
/**
 * @brief constraint approximation with multipliers (and slack variables)
 */
struct constr_impl : public approx {
    constr_impl(const std::string &name, size_t dim, field_t field,
                approx_order order = approx_order::first,
                bool enable_slack = false)
        : approx(name, dim, field, order) {
        assert(field == __dyn || magic_enum::enum_name(field).find(
                                     "cstr") != std::string::npos);
        /// @todo : make dual variables
        // if (enable_slack) {
        // }
        value = [this](auto &d) { approx::value_impl(d); };
        jacobian = [this](auto &d) { approx::jacobian_impl(d); };
        hessian = [this](auto &d) { approx::hessian_impl(d); };
    }

    constr_impl(constr_impl &&rhs)
        : approx(std::move(rhs)),
          value(std::move(rhs.value)),
          jacobian(std::move(rhs.jacobian)),
          hessian(std::move(rhs.hessian)) {}

    /**
     * @brief wrapped data maker for constr
     *
     * @param primal ptr to primal data
     * @param raw ptr to approximation data
     * @return sparse_approx_data_ptr_t
     */
    sparse_approx_data_ptr_t make_data(sym_data *primal, approx_storage *raw) override {
        return constr_data_ptr_t(
            new constr_data(raw, std::move(*approx::make_data(primal, raw)), this));
    }

  private:
    void value_impl(sparse_approx_data &data) override final { value(data); }
    void jacobian_impl(sparse_approx_data &data) override final;
    void hessian_impl(sparse_approx_data &data) override final;

  public:
    std::function<void(sparse_approx_data &)> value;
    std::function<void(sparse_approx_data &)> jacobian;
    std::function<void(constr_data &)> hessian;
};
def_ptr(constr_impl);
/**
 * @brief wrapper of constr_impl, in fact a pointer
 *
 */
struct constr : public constr_impl_ptr_t {
    constr(const std::string &name, size_t dim, field_t field,
           approx_order order = approx_order::first,
           bool enable_slack = false)
        : constr_impl_ptr_t(new constr_impl(name, dim, field, order)) {
    }
    constr(constr_impl &&impl) : constr_impl_ptr_t(new constr_impl(std::move(impl))) {}
    constr(const constr &rhs) = default;
};
} // namespace atri

#endif // ATRI_CONSTR_HPP