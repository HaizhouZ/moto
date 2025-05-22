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
                approx_order order = approx_order::first)
        : approx(name, dim, field, order) {
        assert(field == __dyn || magic_enum::enum_name(field).find(
                                     "cstr") != std::string::npos);
        /// @todo : make dual variables
        // if (enable_slack) {
        // }
    }

    constr_impl(constr_impl &&rhs)
        : approx(std::move(rhs)) {}

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
};
def_ptr(constr_impl);
/**
 * @brief wrapper of constr_impl, in fact a pointer
 *
 */
struct constr : public constr_impl_ptr_t {
    constr(const std::string &name, size_t dim, field_t field,
           approx_order order = approx_order::first)
        : constr_impl_ptr_t(new constr_impl(name, dim, field, order)) {
    }
    constr() = default;
    constr(constr_impl &&impl) : constr_impl_ptr_t(new constr_impl(std::move(impl))) {}
    constr(constr_impl *impl) : constr_impl_ptr_t(impl) {}
    constr(const constr &rhs) = default;

    constr(const std::string &name, std::vector<sym> in_args,
           const cs::SX &expr, field_t type, approx_order order) {
        auto p = new constr_impl(name, expr.rows(), type, order);
        p->add_arguments(in_args);
        std::vector<cs::SX> sx_args(in_args.begin(), in_args.end());
        p->value = [=](sparse_approx_data &d) {
            static auto func = cs::Function(name, sx_args, {expr});
            std::vector<const double *> in_;
            for (auto &in : d.in_args_) {
                in_.push_back(in.data());
            }
            func(in_, std::vector<double *>{d.v_.data()});
        };
        std::vector<cs::Function> jac_funcs;
        std::vector<cs::SX> jac_out;
        for (size_t i = 0; i < sx_args.size(); ++i) {
            for (size_t j = 0; j < sx_args[i].rows(); j++) {
                if (in_args[i]->field_ < field::num_prim)
                    jac_out.push_back(cs::SX::jacobian(expr, sx_args[i](j)));
            }
        }
        cs::Function jac_func(name + "_jac", sx_args, jac_out);
        p->jacobian = [=](sparse_approx_data &d) {
            std::vector<const double *> in_;
            for (auto &in : d.in_args_) {
                in_.push_back(in.data());
            }
            // Flatten jacobian storage for each argument
            std::vector<double *> jac_flat;
            size_t idx_arg = 0;
            for (auto jac : d.jac_) {
                if (d.f_->in_args()[idx_arg]->field_ < field::num_prim) {
                    for (size_t i_row : range(jac.cols())) {
                        jac_flat.push_back(jac.col(i_row).data());
                    }
                }
                idx_arg++;
            }
            jac_func(in_, jac_flat);
        };
        reset(p);
    }
};
} // namespace atri

#endif // ATRI_CONSTR_HPP