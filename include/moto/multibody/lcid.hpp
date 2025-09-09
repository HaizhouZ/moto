#ifndef MOTO_MULTIBODY_LCID_HPP
#define MOTO_MULTIBODY_LCID_HPP

#include <moto/multibody/state.hpp>
#include <moto/ocp/constr.hpp>
#include <moto/ocp/impl/custom_func.hpp>
#include <moto/utils/blasfeo_factorizer/blasfeo_llt.hpp>

namespace moto {
namespace multibody {
/// @brief lifted contact inverse dynamics
struct lcid : public generic_custom_func {
    struct constr_with_state {
        std::vector<state> s;
        std::vector<var> f;
        constr c;
    };

    struct MOTO_ALIGN_NO_SHARING data : public func_arg_map {
        using base = func_arg_map;
        using base::base;
        sparse_mat Minv_;
        sparse_mat Jc_T_;
        matrix osim_inv_;
        matrix Jc_Minv_;    // Jc * Minv
        matrix G_a_;        // acceleration part of G
        matrix G_off_diag_; // off-diagonal part of G, osim_inv * Jc * Minv
        matrix G_;          // full G matrix
        sparse_mat tq_S_;   // torque selection matrix
    };


    size_t ntq_ = 0, nv_ = 0, nc_ = 0;
    bool has_timestep_ = false;

    lcid(const std::string &name);

    // void add_euler(const multibody::euler &e) { eulers_.emplace_back(e); }
    void add_kin_constr(const std::vector<state> &s, var_inarg_list fs, const constr &c) {
        kin_constr_.emplace_back(constr_with_state{.s = s, .c = c});
        kin_constr_.back().f = var_list(fs);
    }
    void add_dyn_constr(const std::vector<state> &s, var_inarg_list fs, const constr &c){
        dyn_constr_.emplace_back(constr_with_state{.s = s, .c = c});
        dyn_constr_.back().f = var_list(fs);
    }
    void compute_osim_inv(data &d);
    void finalize_impl() override;
    std::vector<constr_with_state> kin_constr_, dyn_constr_;
};
} // namespace multibody
} // namespace moto

#endif