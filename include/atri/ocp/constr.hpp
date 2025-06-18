#ifndef ATRI_CONSTR_HPP
#define ATRI_CONSTR_HPP

#include <atri/ocp/func.hpp>

namespace atri {
struct constr_impl; // fwd
/**
 * @brief constraint data
 * derived from sparse_approx_data with multipler and vjp (for cost) mapping in addition
 */
struct constr_data : public sparse_approx_data {
    /// @todo: add this to raw
    // vector_ref slack_;
    double *merit_;
    vector_ref multiplier_;
    std::vector<row_vector_ref> vjp_;
    constr_data(approx_storage *raw, sparse_approx_data &&d, constr_impl *cstr);
};
def_unique_ptr(constr_data);
/**
 * @brief constraint approximation with multipliers (and slack variables)
 */
class constr_impl : public func {
  private:
    void value_impl(sparse_approx_data &data) override final;
    void jacobian_impl(sparse_approx_data &data) override final;
    bool finalize_impl() override;

  public:
    constr_impl(const std::string &name, approx_order order = approx_order::first, size_t dim = 0, field_t field = __undefined)
        : func(name, order, dim, field) {
        /// @todo : make dual variables
    }
    constr_impl(const constr_impl &rhs) = delete;
    constr_impl(constr_impl &&rhs)
        : func(std::move(rhs)) {}

    /**
     * @brief wrapped data maker for constr
     *
     * @param primal ptr to primal data
     * @param raw ptr to approximation data
     * @return sparse_approx_data_ptr_t
     */
    sparse_approx_data_ptr_t make_approx_data_mapping(sym_data *primal, approx_storage *raw, shared_data* shared) override {
        return constr_data_ptr_t(
            new constr_data(raw, std::move(*func::make_approx_data_mapping(primal, raw, shared)), this));
    }
};
def_ptr(constr_impl);
/**
 * @brief wrapper of constr_impl, in fact a pointer
 *
 */
struct constr : public constr_impl_ptr_t {
    constr(const std::string &name, approx_order order = approx_order::first, size_t dim = 0, field_t field = __undefined)
        : constr_impl_ptr_t(new constr_impl(name, order, dim, field)) {
    }
    constr() = default;
    constr(constr_impl &&impl) : constr_impl_ptr_t(new constr_impl(std::move(impl))) {}
    constr(constr_impl *impl) : constr_impl_ptr_t(impl) {}
    constr(const constr &rhs) = default;
};
} // namespace atri

#endif // ATRI_CONSTR_HPP