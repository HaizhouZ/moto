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
    double *merit_;
    vector_ref multiplier_;
    std::vector<row_vector_ref> vjp_;
    constr_data(approx_storage *raw, sparse_approx_data &&d, constr_impl *cstr);
};
def_unique_ptr(constr_data);
/**
 * @brief constraint approximation with multipliers (and slack variables)
 */
class constr_impl : public approx {
  private:
    void value_impl(sparse_approx_data &data) override final;
    void jacobian_impl(sparse_approx_data &data) override final;
    void hessian_impl(sparse_approx_data &data) override final;
    bool finalize_impl() override {
        if (field_ == __undefined) {
            bool has_[3] = {false, false, false};
            for (const auto &arg : in_args_) {
                if (arg->field_ <= __y)
                    has_[arg->field_] = true;
            }
            // make this long enough so that people will not easily remove the const :D
            auto &field = *const_cast<field_t *>(&field_);
            if (has_[__u] && !has_[__y])
                field = __eq_cstr_c;
            else if (has_[__u] && has_[__x] && has_[__y])
                field = __dyn;
            else if (!has_[__u] && !has_[__x] && has_[__y])
                field = __eq_cstr_s;
            else
                throw std::runtime_error(fmt::format("unsupported constr type has_x: {}, has_u: {}, has_y: {}", has_[__x], has_[__u], has_[__y]));
            if (field_ == __eq_cstr_s) {
                // do in_arg substitute
                try {
                    for (auto &arg : in_args_) {
                        if (arg->field_ == __x) {
                            arg = sym(expr_index::get(arg->name_ + "_nxt"));
                        }
                    }
                } catch (const std::exception &ex) {
                    fmt::print("substitute exception");
                    throw;
                }
            }
        }
        assert(field_ == __dyn || magic_enum::enum_name(field_).find(
                                      "cstr") != std::string::npos);
        return true;
    }

  public:
    constr_impl(const std::string &name, approx_order order = approx_order::first, size_t dim = 0, field_t field = __undefined)
        : approx(name, order, dim, field) {
        /// @todo : make dual variables
        // if (enable_slack) {
        // }
        value = [this](auto &d) { approx::value_impl(d); };
        jacobian = [this](auto &d) { approx::jacobian_impl(d); };
        hessian = [this](auto &d) { approx::hessian_impl(d); };
    }
    constr_impl(const constr_impl &rhs) = delete;
    constr_impl(constr_impl &&rhs)
        : approx(std::move(rhs)),
          value(std::move(rhs.value)),
          jacobian(std::move(rhs.jacobian)),
          hessian(std::move(rhs.hessian)) {}

    void load_external(const std::string &path = "gen");

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
    constr(const std::string &name, approx_order order = approx_order::first, size_t dim = 0, field_t field = __undefined)
        : constr_impl_ptr_t(new constr_impl(name, order, dim, field)) {
    }
    constr() = default;
    constr(constr_impl &&impl) : constr_impl_ptr_t(new constr_impl(std::move(impl))) {}
    constr(constr_impl *impl) : constr_impl_ptr_t(impl) {}
    constr(const constr &rhs) = default;
};

// struct casadi_approx {
//     casadi_approx(const std::string &name, std::vector<sym> in_args,
//                   const cs::SX &expr, field_t type, approx_order order) {

//     }
// };

} // namespace atri

#endif // ATRI_CONSTR_HPP