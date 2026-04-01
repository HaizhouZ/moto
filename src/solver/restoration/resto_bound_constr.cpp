#include <moto/solver/restoration/resto_bound_constr.hpp>
#include <moto/ocp/problem.hpp>

namespace moto {
namespace solver {
namespace {
generic_constr make_resto_bound_generic(resto_bound_type type, size_t dim, const std::string &name) {
    auto s = sym::symbol(name + "_storage", dim, __s);
    cs::SX out = (type == resto_bound_type::positive) ? -static_cast<sym &>(*s)
                                                      :  static_cast<sym &>(*s);
    return generic_constr(name, var_inarg_list{s}, out, approx_order::first, __ineq_x);
}
} // namespace

resto_bound_constr::approx_data::approx_data(ipm_constr::approx_data &&rhs)
    : ipm_constr::approx_data(std::move(rhs)),
      storage_(problem()->extract(primal_->value_[__s], func_.in_args()[0])),
      storage_backup_(storage_.size()) {
    storage_backup_.setZero();
}

resto_bound_constr::resto_bound_constr(resto_bound_type type, size_t dim, const std::string &name)
    : ipm_constr(make_resto_bound_generic(type, dim, name)) {
    bound_type = type;
}

void resto_bound_constr::initialize(data_map_t &data) const {
    auto &d = data.as<approx_data>();
    ipm_constr::initialize(d);
    d.storage_ = d.slack_;
    d.storage_backup_ = d.storage_;
}

void resto_bound_constr::backup_trial_state(data_map_t &data) const {
    auto &d = data.as<approx_data>();
    ipm_constr::backup_trial_state(d);
    d.storage_backup_ = d.storage_;
}

void resto_bound_constr::restore_trial_state(data_map_t &data) const {
    auto &d = data.as<approx_data>();
    ipm_constr::restore_trial_state(d);
    d.storage_ = d.storage_backup_;
}

void resto_bound_constr::apply_affine_step(data_map_t &data, workspace_data *cfg) const {
    auto &d = data.as<approx_data>();
    ipm_constr::apply_affine_step(d, cfg);
    d.storage_ = d.slack_;
}

} // namespace solver
} // namespace moto
