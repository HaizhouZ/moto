#include <moto/ocp/dynamics.hpp>
#include <moto/ocp/problem.hpp>

namespace moto {

generic_dynamics::approx_data::approx_data(base::approx_data &&rhs)
    : base::approx_data(std::move(rhs)),
      proj_f_res_(problem()->extract(merit_data_->dynamics_data_.proj_f_res_, func_)) {
    approx_ = &merit_data_->approx_[__dyn];
    dyn_proj_ = &merit_data_->dynamics_data_;
}
} // namespace moto