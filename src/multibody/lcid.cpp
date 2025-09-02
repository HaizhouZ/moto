#include <moto/multibody/lcid.hpp>
#include <moto/ocp/problem.hpp>

namespace moto {
namespace multibody {
lcid::lcid(const std::string &name) : generic_custom_func(name, approx_order::none, 0, __usr_func) {
    create_custom_data = [this](sym_data &primal, shared_data &shared) {
        auto ptr = std::make_unique<data>(primal, shared, *this);
        ptr->Minv_.resize(nv_, nv_);
        ptr->Jc_Minv_.resize(nc_, nv_);
        ptr->osim_inv_.resize(nc_, nc_);
        ptr->G_off_diag_.resize(nv_, nv_);
        ptr->G_a_.resize(nv_, nv_);
        ptr->tq_S_.resize(nv_, nv_);
        ptr->G_.resize(nv_ + nc_, nv_ + nc_);
        size_t r_st = 0;
        for (const auto &d : dyn_constr_) {
            auto &s = d.s[0]; // assume every dynamic constraint has only one state
            size_t indent = (s.root_type != root_joint_t::fixed ? 6 : 0);
            ptr->tq_S_.insert<sparsity::eye>(r_st + indent, 0, s.v->dim() - indent);
            ptr->Minv_.insert<sparsity::dense>(r_st, r_st, s.v->dim());
            r_st += s.v->dim();
        }
        auto &prob = *ptr->problem();
        r_st = 0;
        for (const auto &d : kin_constr_) {
            for (const auto &s : d.s) {
                size_t c_st = prob.get_expr_start_tangent(s.q); // assume q is always the first
                ptr->Jc_T_.insert<sparsity::dense>(r_st, prob.get_expr_start_tangent(s.q), s.q->tdim());
                ptr->Jc_T_.insert<sparsity::dense>(r_st, prob.get_expr_start_tangent(s.v), s.v->dim());
            }
            r_st += d.c->dim();
        }
        return ptr;
    };
}
void lcid::finalize_impl() {
    // reorder deps, add dynamics first
    for (const auto &d : dyn_constr_) {
        add_dep(d.c);
        auto &s = d.s[0]; // assume every dynamic constraint has only one state
        if (s.dt->field() == __u)
            has_timestep_ = true;
        nv_ += s.v->dim();
        ntq_ += nv_ - (s.root_type != root_joint_t::fixed ? 6 : 0);
    }
    for (const auto &c : kin_constr_) {
        add_dep(c.c);
        nc_ += c.c->dim();
    }
    generic_custom_func::finalize_impl();
}
void lcid::compute_osim_inv(data &d) {
    for (size_t i = 0; i < M_llt_.size(); i++) {
        auto &llt = M_llt_[i];
        llt.compute(d.Minv_.dense_panels_[0].data_);
        llt.get_inverse(d.Minv_.dense_panels_[0].data_);
    }
    d.Jc_Minv_.resize(d.Jc_T_.rows(), d.Minv_.cols());
    d.Jc_Minv_.setZero();
    d.Jc_T_.T_times(d.Minv_, d.Jc_Minv_);
    d.osim_inv_.setZero();
    d.Jc_T_.right_times(d.Jc_Minv_, d.osim_inv_);
    llt_.compute(d.osim_inv_);
    llt_.get_inverse(d.osim_inv_);
    d.G_off_diag_.noalias() = d.osim_inv_ * d.Jc_Minv_;
    d.G_a_.setZero();
    thread_local utils::buffer_tpl<matrix> tmp;
    tmp.resize(d.Minv_.rows(), d.G_off_diag_.cols());
    tmp.data_.setZero();
    d.Jc_T_.times<false>(d.G_off_diag_, tmp.data_);
    tmp.data_.diagonal().array() += 1.0;
    d.Minv_.times(tmp.data_, d.G_a_);
    d.G_.topLeftCorner(nv_, nv_) = d.G_a_;
    d.G_.bottomLeftCorner(nc_, nv_) = d.Jc_Minv_;
    d.G_.topRightCorner(nv_, nc_) = d.Jc_Minv_.transpose();
    d.G_.bottomRightCorner(nc_, nc_) = d.osim_inv_;
}
} // namespace multibody
} // namespace moto
