#ifndef MOTO_OCP_EULER_DYNAMICS_HPP
#define MOTO_OCP_EULER_DYNAMICS_HPP

#include <moto/ocp/dynamics.hpp>

namespace moto {

struct explicit_euler;

class explicit_euler_impl : public generic_dynamics {
  private:
    scalar_t dt_ = 0.0;         ///< time step
    bool has_1st_ord_ = false;  ///< true if first order dynamics are present
    bool has_2nd_ord_ = false;  ///< true if second order dynamics are present
    bool has_timestep_ = false; ///< true if timestep variable is present
    struct second_order {
        size_t dim_ = 0; ///< dimension of the state
        var_list pos_x_; ///< position variables
        var_list pos_y_; ///< position variables
        var_list vel_x_; ///< velocity variables
        var_list vel_y_; ///< velocity variables
        var_list acc_u_; ///< acceleration variables)
        void add(const var &r, const var &rn, const var &v, const var &vn, const var &a) {
            pos_x_.push_back(r);
            pos_y_.push_back(rn);
            vel_x_.push_back(v);
            vel_y_.push_back(vn);
            acc_u_.push_back(a);
            dim_ += r->dim();
        }
    } sec_ord_var_; ///< second order variables
    struct first_order {
        size_t dim_ = 0; ///< dimension of the state
        var_list pos_x_; ///< position variables in state field
        var_list pos_y_; ///< position variables in state field
        var_list vel_u_; ///< velocity variables in control field
        void add(const var &r, const var &rn, const var &v) {
            pos_x_.push_back(r);
            pos_y_.push_back(rn);
            vel_u_.push_back(v);
            dim_ += r->dim();
        }
    } first_ord_var_;  ///< first order variables
    var timestep_var_; ///< time variable
    virtual void finalize_impl() override;
    friend struct explicit_euler; ///< allow explicit_euler to access private members

  public:
    struct approx_data : public generic_dynamics::approx_data {
        aligned_vector_map_t f_x_off_diag_, proj_f_x_off_diag_; ///< x field jacobian
        aligned_vector_map_t f_u_, proj_f_u_;                   ///< x field projection jacobian
        aligned_vector_map_t f_dt_, proj_f_dt_;                 ///< time step jacobian
        approx_data(generic_dynamics::approx_data &&rhs);
    };
    func_approx_data_ptr_t create_approx_data(sym_data &primal,
                                              merit_data &raw,
                                              shared_data &shared) const override {
        return func_approx_data_ptr_t(make_approx<explicit_euler_impl>(primal, raw, shared));
    }
    void value_impl(func_approx_data &data) const override;
    void jacobian_impl(func_approx_data &data) const override;

    void compute_project_derivatives(func_approx_data &data) const override;
};

struct explicit_euler : public func {
    explicit_euler_impl *operator->() const {
        return static_cast<explicit_euler_impl *>(func::operator->());
    }
    /// @brief add position variables
    auto create_2nd_ord_vars(const std::string &name, size_t dim) {
        auto [r, rn] = sym::states(name, dim);
        auto [v, vn] = sym::states(name + "_vel", dim);
        auto a = sym::inputs(name + "_acc", dim);
        (*this)->sec_ord_var_.add(r, rn, v, vn, a);
        return std::tuple{std::move(r), std::move(rn), std::move(v), std::move(vn), std::move(a)};
    }
    /// @brief create first order variables
    auto create_1st_ord_vars(const std::string &name, size_t dim) {
        auto [r, rn] = sym::states(name, dim);
        auto v = sym::inputs(name + "_vel", dim);
        (*this)->first_ord_var_.add(r, rn, v);
        return std::tuple{std::move(r), std::move(rn), std::move(v)};
    }
    void add_dt(scalar_t dt) { (*this)->dt_ = dt; }             ///< set time step
    void add_dt(const var &dt) { (*this)->timestep_var_ = dt; } ///< set time variable
};

} // namespace moto

#endif