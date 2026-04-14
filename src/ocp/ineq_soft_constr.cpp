#include <moto/ocp/ineq_constr.hpp>
#include <moto/ocp/problem.hpp>

namespace moto {

void soft_constr::propagate_jacobian(func_approx_data &data, const vector_const_ref &residual, scalar_t scale) const {
    auto &d = data.as<data_map_t>();
    if (residual.size() == 0 || scale == 0.) {
        return;
    }
    for (size_t arg_idx = 0; arg_idx < d.lag_jac_corr_.size(); ++arg_idx) {
        if (d.has_jacobian_block(arg_idx)) {
            d.lag_jac_corr_[arg_idx].noalias() += (scale * residual).transpose() * d.jac_[arg_idx];
        }
    }
}

void soft_constr::propagate_hessian(func_approx_data &data, const vector_const_ref &diag_scaling, scalar_t scale) const {
    auto &d = data.as<data_map_t>();
    if (diag_scaling.size() == 0 || scale == 0.) {
        return;
    }
    size_t outer_idx = 0;
    for (auto &outer : d.lag_hess_) {
        size_t inner_idx = 0;
        if (outer.size()) {
            for (auto &inner : outer) {
                if (inner.size() != 0) {
                    inner.noalias() +=
                        scale * (d.jac_[outer_idx].transpose() * diag_scaling.asDiagonal() * d.jac_[inner_idx]);
                }
                ++inner_idx;
            }
        }
        ++outer_idx;
    }
}

void soft_constr::propagate_hessian(func_approx_data &data, scalar_t scale) const {
    auto &d = data.as<data_map_t>();
    if (scale == 0.) {
        return;
    }
    size_t outer_idx = 0;
    for (auto &outer : d.lag_hess_) {
        size_t inner_idx = 0;
        if (outer.size()) {
            for (auto &inner : outer) {
                if (inner.size() != 0) {
                    inner.noalias() += scale * (d.jac_[outer_idx].transpose() * d.jac_[inner_idx]);
                }
                ++inner_idx;
            }
        }
        ++outer_idx;
    }
}

void soft_constr::setup_hess() {
    if (!in_field(field_, ineq_soft_constr_fields)) {
        return;
    }
    for (size_t i : range(jac_sp_.size())) {
        for (size_t j : range(jac_sp_.size())) {
            const auto lhs_jac_sp = jac_sp_[i];
            const auto rhs_jac_sp = jac_sp_[j];
            auto &hess = hess_sp_[i][j];
            hess.row_offset = lhs_jac_sp.col_offset;
            hess.col_offset = rhs_jac_sp.col_offset;
            hess.rows = lhs_jac_sp.cols;
            hess.cols = rhs_jac_sp.cols;
            if (lhs_jac_sp.pattern == sparsity::unknown || rhs_jac_sp.pattern == sparsity::unknown) {
                hess.pattern = sparsity::unknown;
            } else if (lhs_jac_sp.pattern == sparsity::dense || rhs_jac_sp.pattern == sparsity::dense) {
                hess.pattern = sparsity::dense;
            } else {
                hess.pattern = sparsity::diag;
            }
        }
    }
}

void soft_constr::load_external_impl(const std::string &path) {
    base::load_external_impl(path);
    setup_hess();
}

soft_constr::approx_data::approx_data(data_base &&rhs)
    : data_base(std::move(rhs)),
      d_multiplier_(nullptr, 0) {
    map_lag_jac_from_raw(lag_data_->lag_jac_corr_, lag_jac_corr_);
}

void soft_constr::finalize_impl() {
    base::finalize_impl();
    if (!skip_field_check && !in_field(field_, soft_constr_fields))
        throw std::runtime_error(fmt::format(
            "Soft constraint {} must have field in soft_constr_fields, but got {}", name_, field::name(field_)));
}
void ineq_constr::finalize_impl() {
    skip_field_check = true;
    base::finalize_impl();
    if (!in_field(field_, ineq_constr_fields))
        throw std::runtime_error(fmt::format(
            "Inequality constraint {} must have field in ineq_constr_fields, but got {}", name_, field::name(field_)));
}

ineq_constr::approx_data::approx_data(approx_data::data_base &&d)
    : base::approx_data(std::move(d)),
      comp_(problem()->extract(lag_data_->comp_[func_.field()], func_)) {
    const auto *constr = dynamic_cast<const ineq_constr *>(&func_);
    const auto *box = constr != nullptr ? constr->box_info() : nullptr;
    if (box != nullptr) {
        box_spec_ = box;
        const auto n = static_cast<Eigen::Index>(box->base_dim);
        for (auto side : box_sides) {
            box_side_[side] = constr->create_side_data();
            box_side_[side]->resize(n);
            if (box->bound_source[side] == ineq_constr::box_bound_source::constant) {
                box_const_[side] = box->bound_constant_value[side];
            }
        }
    }
}

void ineq_constr::value_impl(func_approx_data &data) const {
    base::value_impl(data);
    auto &d = data.as<approx_data>();
    if (d.boxed()) {
        const auto &box = d.require_box_spec("ineq_constr::value_impl");
        for (auto side : box_sides) {
            auto &pair = *d.box_side_[side];
            const auto &mask = box.present_mask[side];
            const scalar_t residual_sign = side == box_side::ub ? scalar_t(1) : scalar_t(-1);
            if (box.bound_source[side] == box_bound_source::constant) {
                pair.residual.array() = mask.select(residual_sign * (d.v_.array() - d.box_const_[side].array()), scalar_t(0));
            } else if (box.bound_source[side] == box_bound_source::in_arg) {
                const auto bound = d[static_cast<const sym &>(box.bound_var[side])];
                pair.residual.array() = mask.select(residual_sign * (d.v_.array() - bound.array()), scalar_t(0));
            } else {
                pair.residual.setZero();
            }
        }
    }
    d.comp_.array() = d.multiplier_.array() * d.v_.array();
} // namespace impl

} // namespace moto
