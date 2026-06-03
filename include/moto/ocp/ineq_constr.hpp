#ifndef MOTO_OCP_IMPL_INEQ_CONSTR_HPP
#define MOTO_OCP_IMPL_INEQ_CONSTR_HPP

#include <moto/core/array.hpp>
#include <moto/ocp/soft_constr.hpp>
#include <algorithm>
#include <limits>

namespace moto {
namespace box_side {
enum side_t : size_t {
    ub = 0,
    lb = 1,
};
}

inline constexpr std::array box_sides{box_side::ub, box_side::lb};
/**
 * @brief inequality constraint interface class
 *
 */
class ineq_constr : public soft_constr {
  private:
    using base = soft_constr;

  public:
    template <typename T>
    using box_side_array = array_type<T, box_sides>;
    using box_bound_t = std::variant<scalar_t, vector, cs::SX>;
    enum class box_bound_source {
        none,
        constant,
        in_arg,
    };
    struct box_spec {
        size_t base_dim = 0;
        box_side_array<Eigen::Array<bool, Eigen::Dynamic, 1>> present_mask;
        box_side_array<bool> has_side = {};
        box_side_array<box_bound_source> bound_source = {};
        box_side_array<var> bound_var = {};
        box_side_array<vector> bound_constant_value = {};
    };
  private:
    std::shared_ptr<box_spec> box_spec_;

  public:
    /**
     * @brief inequality constraint approximation map with the complementarity residual map
     *
     */
    struct approx_data : public base::approx_data {
        struct box_pair_runtime {
            vector residual;
            vector slack;
            vector slack_backup;
            vector d_slack;
            vector multiplier;
            vector multiplier_backup;
            vector d_multiplier;
            virtual ~box_pair_runtime() = default;

            virtual void resize(Eigen::Index n) {
                residual.setZero(n);
                slack.setZero(n);
                slack_backup.setZero(n);
                d_slack.setZero(n);
                multiplier.setZero(n);
                multiplier_backup.setZero(n);
                d_multiplier.setZero(n);
            }
        };

        vector_ref comp_;
        box_side_array<vector> box_const_ = {};
        const box_spec *box_spec_ = nullptr;
        box_side_array<std::unique_ptr<box_pair_runtime>> box_side_;
        approx_data(data_base &&d);

        const box_spec &require_box_spec(std::string_view where) const {
            if (box_spec_ == nullptr) {
                throw std::runtime_error(fmt::format("{} requires boxed ineq runtime metadata for {}", where, func_.name()));
            }
            return *box_spec_;
        }

        bool boxed() const noexcept { return box_spec_ != nullptr; }
    };

  protected:
    /// @brief finalize the inequality constraint, will be called upon added to a problem
    void finalize_impl() override;
    /// @brief evaluate the value of the constraint and compute the complementarity residual
    void value_impl(func_approx_data &data) const override;

  public:
    using base::base;
    ineq_constr(const std::string &name, approx_order order, size_t dim, field_t field = field_t::__undefined)
        : base(name, order, dim, field) {
        field_hint().is_eq = false;
    }
    ineq_constr(const std::string &name,
                const var_inarg_list &args,
                const cs::SX &out,
                approx_order order,
                field_t field = field_t::__undefined)
        : base(name, args, out, order, field) {
        field_hint().is_eq = false;
    }
    ineq_constr(generic_constr &&rhs) : base(std::move(rhs)) {
        if (auto *src = dynamic_cast<ineq_constr *>(&rhs); src != nullptr) {
            box_spec_ = src->box_spec_;
        }
        field_hint().is_eq = false; ///< set the field hint to inequality
    } ///< move constructor from generic_constr
    void set_box_info(std::shared_ptr<box_spec> box) { box_spec_ = std::move(box); }
    const box_spec *box_info() const noexcept { return box_spec_.get(); }
    virtual std::unique_ptr<approx_data::box_pair_runtime> create_side_data() const {
        return std::make_unique<approx_data::box_pair_runtime>();
    }
    void initialize(data_map_t &) const override {}
    void finalize_newton_step(data_map_t &) const override {}
    void apply_affine_step(data_map_t &, workspace_data *) const override {}
    virtual void restoration_commit_dual_step(data_map_t &data, scalar_t alpha_dual) const {
        data.multiplier_.noalias() += alpha_dual * data.d_multiplier_;
    }
    virtual void restoration_reset_bound_multipliers(data_map_t &data) const {
        data.multiplier_.setConstant(1.0);
    }
    void synthesize_upper_half_box_info_if_missing() {
        if (box_spec_ != nullptr) {
            return;
        }
        auto spec = std::make_shared<box_spec>();
        const auto dim = static_cast<Eigen::Index>(this->dim());
        spec->base_dim = static_cast<size_t>(dim);
        for (auto side : box_sides) {
            spec->present_mask[side] = Eigen::Array<bool, Eigen::Dynamic, 1>::Constant(dim, side == box_side::ub);
            spec->has_side[side] = side == box_side::ub;
            spec->bound_source[side] = box_bound_source::constant;
            spec->bound_constant_value[side] =
                side == box_side::ub ? vector::Zero(dim) : vector::Constant(dim, -std::numeric_limits<scalar_t>::infinity());
        }
        box_spec_ = std::move(spec);
    }
    static constr create(const std::string &name,
                         const var_inarg_list &args,
                         const cs::SX &out,
                         approx_order order = approx_order::first,
                         field_t field = field_t::__undefined);
    static constr create(const std::string &name,
                         approx_order order = approx_order::first,
                         size_t dim = dim_tbd,
                         field_t field = field_t::__undefined);
    static constr create(const std::string &name,
                         const var_inarg_list &args,
                         const cs::SX &out,
                         const box_bound_t &lb,
                         const box_bound_t &ub,
                         approx_order order = approx_order::first,
                         field_t field = field_t::__undefined);
    /***
     * @brief make approximation data for the inequality constraint, will use default @ref data_type
     */
    func_approx_data_ptr_t create_approx_data(sym_data &primal, lag_data &raw, shared_data &shared) const override {
        return func_approx_data_ptr_t(make_approx<ineq_constr>(primal, raw, shared));
    }
};

} // namespace moto

#endif // MOTO_OCP_IMPL_INEQ_CONSTR_HPP
