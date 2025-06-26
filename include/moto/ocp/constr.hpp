#ifndef MOTO_CONSTR_HPP
#define MOTO_CONSTR_HPP

#include <moto/ocp/func.hpp>

namespace moto {
struct constr_impl; // fwd
/**
 * @brief constraint data
 * derived from sp_approx_map with multipler and vjp (for cost) mapping in addition
 */
struct constr_data : public sp_approx_map {
    /// @todo: add this to raw
    // vector_ref slack_;
    double *merit_;
    vector_ref multiplier_;
    std::vector<row_vector_ref> vjp_;
    constr_data(approx_storage &raw, sp_approx_map &&d, constr_impl *cstr);
};
def_unique_ptr(constr_data);
/**
 * @brief constraint approximation with multipliers (and slack variables)
 */
class constr_impl : public func_impl {
  private:
    void value_impl(sp_approx_map &data) override final;
    void jacobian_impl(sp_approx_map &data) override final;
    bool finalize_impl() override;

  public:
    constr_impl(const std::string &name, approx_order order = approx_order::first, size_t dim = 0, field_t field = __undefined)
        : func_impl(name, order, dim, field) {
        /// @todo : make dual variables
    }

    /**
     * @brief wrapped data maker for constr
     *
     * @param primal ptr to primal data
     * @param raw ptr to approximation data
     * @param shared ptr to shared data
     * @return sp_approx_map_ptr_t
     */
    sp_approx_map_ptr_t make_approx_data_mapping(sym_data &primal, approx_storage &raw, shared_data &shared) override {
        return constr_data_ptr_t(
            new constr_data(raw, std::move(*func_impl::make_approx_data_mapping(primal, raw, shared)), this));
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
    constr(constr_impl *impl) : constr_impl_ptr_t(impl) {}
    constr(const constr &rhs) = default;
};
} // namespace moto

#endif // MOTO_CONSTR_HPP