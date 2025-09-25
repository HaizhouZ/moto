#include <Eigen/LU>
#include <moto/core/external_function.hpp>
#include <moto/ocp/dynamics/euler.hpp>
#include <moto/ocp/dynamics/euler_data.hpp>
#include <moto/utils/codegen.hpp>

namespace moto {
namespace multibody {
void euler::jac_block::setup(const cs::SX &mat, size_t c_offset) {
    dense = cs::SX::sparsify(mat);
    assert(dense.is_square() && "the jacobian must be square");
    size_t nnz_total = dense.nnz();
    if (nnz_total == 0) {
        dense.clear();
        return;
    }
    diag = cs::SX::sparsify(cs::SX::diag(dense));
    size_t diag_nnz = diag.nnz();
    dense = cs::SX::sparsify(dense - cs::SX::diag(diag));
    // std::cout << dense << std::endl;
    size_t dense_nnz = dense.nnz();
    assert(dense.nnz() + diag.nnz() == nnz_total && "sparsify error");
    empty = dense.nnz() == 0 && diag.nnz() == 0; /// @todo append dense to diag
    has_block = dense.nnz() > 0;
    if (has_block) {
        std::vector<casadi_int> rows(dense_nnz), cols(dense_nnz);
        dense.sparsity().get_triplet(rows, cols);
        auto [minr, maxr] = std::minmax_element(rows.begin(), rows.end());
        auto [minc, maxc] = std::minmax_element(cols.begin(), cols.end());
        dense_block.r_st = *minr;
        dense_block.nrow = *maxr - *minr + 1;
        dense_block.c_st = *minc;
        dense_block.ncol = *maxc - *minc + 1;
        for (size_t i = 0; i < dense_block.nrow; i++) {
            size_t idx = dense_block.r_st + i;
            dense(idx, idx) = diag(idx);
            diag(idx) = 0.;
        }
        dense = dense(cs::Slice(dense_block.r_st, dense_block.r_st + dense_block.nrow),
                      cs::Slice(dense_block.c_st, dense_block.c_st + dense_block.ncol));
        // dense_block.c_st += c_offset;
        dense_nnz = dense.nnz();
    }
}
euler::euler(const std::string &name,
             const state &s, cs::SX pos_step,
             cs::SX pos_diff, binary_jac_t dpos_diff, cs::SX pos_int, binary_jac_t dpos_int)
    : base(name,
           {s.q, s.v, s.q->next(), s.v->next(), s.a, s.dt},
           cs::SX::vertcat({pos_diff, v->next() - (v + a * dt)}),
           approx_order::first, __dyn),
      state(s),
      pos_integrate_(pos_int), pos_diff_(pos_diff), pos_step_(pos_step),
      dpos_diff_(dpos_diff), dpos_int_(dpos_int) {
    q->tdim() = pos_diff_.size1();
    q->next()->tdim() = pos_diff_.size1();
    assert(pos_integrate_.size1() == q->dim() && "the position step must have the same dimension as q");
    assert(v->dim() == q->tdim() && "the velocity dimension must match the tangent space dimension of q");
}
void euler::load_external_impl(const std::string &path) {
    generic_func::load_external_impl(path);
    auto f = ext_func(gen_.task_->extra_task->func_name);
    q->integrator() = [f_ = std::move(f), nv = v->dim()](vector_ref x, vector_ref dx, vector_ref out, scalar_t alpha) {
        std::vector<vector_ref> args = {x, dx, mapped_vector(&alpha, 1)};
        f_.invoke(args, out);                // for pos
        out.tail(nv) += dx.tail(nv) * alpha; // for vel
    };
}
void euler::finalize_impl() {
    pos_diff_jac_.dqn.setup(dpos_diff_[0]);
    pos_diff_jac_.dq.setup(cs::SX::mtimes(dpos_diff_[1], dpos_int_[0]));
    auto dpos_diff_dstep = cs::SX::mtimes(dpos_diff_[1], dpos_int_[1]);
    pos_diff_jac_.dvn.setup(cs::SX::mtimes(dpos_diff_dstep, cs::SX::jacobian(pos_step_, v->next())), q->tdim());
    pos_diff_jac_.dv.setup(cs::SX::mtimes(dpos_diff_dstep, cs::SX::jacobian(pos_step_, v)), q->tdim());
    // setup the projection jacobian
    // the inverse of F_y is the same as F_y, maybe buggy if the dqn and dq are not consistent
    if (pos_diff_jac_.dqn.has_block)
        pos_diff_proj_jac_.dq = pos_diff_jac_.dqn;
    if (pos_diff_jac_.dq.has_block)
        pos_diff_proj_jac_.dq = pos_diff_jac_.dq;
    if (pos_diff_jac_.dvn.has_block)
        pos_diff_proj_jac_.dv = pos_diff_jac_.dvn;
    if (pos_diff_jac_.dv.has_block)
        pos_diff_proj_jac_.dv = pos_diff_jac_.dv;
    // setup proj input jacobian
    if (pos_diff_jac_.dvn.has_block) {
        pos_diff_proj_jac_.da = pos_diff_jac_.dvn;
    }
    // these sparsity are enough to setup the spmat
    if (!pos_diff_jac_.dvn.empty) {
        if (pos_diff_jac_.dv.empty) {
            v_int_type_ = v_int_type::_implicit;
        } else {
            v_int_type_ = v_int_type::_mid_point;
        }
    }
    // generate the casadi integrator
    utils::cs_codegen::task integrator_task;
    integrator_task.func_name = name_ + "_int";
    integrator_task.sx_inputs = {q, v, dt}; // v here just represent the step in the integrator
    integrator_task.sx_output = pos_integrate_;
    integrator_task.keep_generated_src = true;
    gen_.task_->extra_task.reset(new utils::cs_codegen::task(std::move(integrator_task)));
    // setup the sparse jacs
    auto &jacs = gen_.task_->sx_outputs;
    jacs.emplace_back(pos_diff_jac_.dqn.dense);
    jacs.emplace_back(pos_diff_jac_.dq.dense);
    jacs.emplace_back(pos_diff_jac_.dvn.dense);
    jacs.emplace_back(pos_diff_jac_.dv.dense);
    jacs.emplace_back(pos_diff_jac_.dvn.diag);
    jacs.emplace_back(pos_diff_jac_.dv.diag);
    if (dt->field() == __u) {
        jacs.emplace_back(cs::SX::jacobian(pos_diff_, dt));
    }
    n_jac_output_ = jacs.size();
    base::finalize_impl();
};

void euler::setup_data(euler_data &data) const {
    size_t i = 0;
    data.jac_.reserve(n_jac_output_);
    data.jac_.clear();
    data.jac_.emplace_back(data.f_y_q_block);
    data.jac_.emplace_back(data.f_x_q_block);
    data.jac_.emplace_back(data.f_y_v_block);
    data.jac_.emplace_back(data.f_x_v_block);
    data.jac_.emplace_back(data.f_y_v_off_diag_);
    data.jac_.emplace_back(data.f_x_v_off_diag_);
    if (dt->field() == __u) {
        data.jac_.emplace_back(data.f_t_q);
    }
}

void euler::jacobian_impl(func_approx_data &data) const {
    base::jacobian_impl(data);
    auto &d = data.as<euler_data>();
    d.f_t_v.array() = -data[a];
}

void euler::compute_project_derivatives(euler_data &data) const {
    if (pos_diff_jac_.dqn.has_block) {
        data.f_y_inv_q_block.noalias() = data.f_y_q_block.inverse();
    }
    if (pos_diff_jac_.dvn.has_block) {
        if (pos_diff_jac_.dqn.has_block) {
            data.f_y_inv_v_block.noalias() = -data.f_y_inv_q_block * data.f_y_v_block;
        } else
            data.f_y_inv_v_block.array() = -data.f_y_v_block.array();
    }
    if (pos_diff_jac_.dqn.has_block) {
        if (pos_diff_jac_.dq.has_block) {
            data.proj_f_x_q_block.noalias() = data.f_y_inv_q_block * data.f_x_q_block;
        } else {
            data.proj_f_x_q_block.array() = -data.f_y_inv_q_block.array();
        }
    }
    if (pos_diff_jac_.dvn.has_block) {
        data.proj_f_x_v_block.array() += -data.f_y_inv_v_block.array(); // f_y diag is -1, cancel with -1 in proj
    }
    if (pos_diff_jac_.dv.has_block) {
        if (pos_diff_jac_.dqn.has_block) {
            data.proj_f_x_v_block.noalias() += data.f_y_inv_q_block * data.f_x_v_block;
        } else
            data.proj_f_x_v_block += data.f_x_v_block;
    }
    /// @todo not safe
    if (pos_diff_proj_jac_.da.has_block) {
        size_t st = pos_diff_proj_jac_.da.dense_block.r_st;
        size_t n = pos_diff_proj_jac_.da.dense_block.nrow;
        data.proj_f_u_q_block.noalias() = data.f_y_inv_v_block * data.f_u_v_diag_.segment(st, n).asDiagonal();
    }
    if (dt->field() == __u) {
        if (pos_diff_jac_.dqn.has_block) {
            size_t st = pos_diff_proj_jac_.dq.dense_block.r_st;
            size_t n = pos_diff_proj_jac_.dq.dense_block.nrow;
            data.proj_f_t_q.segment(st, n).noalias() += data.f_y_inv_q_block * data.f_t_q.segment(st, n);
        }
        if (pos_diff_jac_.dvn.has_block) {
            size_t st = pos_diff_proj_jac_.dv.dense_block.r_st;
            size_t n = pos_diff_proj_jac_.dv.dense_block.nrow;
            data.proj_f_t_q.segment(st, n).noalias() += data.f_y_inv_v_block * data.f_t_v.segment(st, n);
        }
    }
}
func euler::share(const std::string &name) const {
    euler e;
    static_cast<generic_func &>(e) = base::share(true, {dt});
    e.pos_diff_jac_ = pos_diff_jac_;
    e.pos_diff_proj_jac_ = pos_diff_proj_jac_;
    e.n_jac_output_ = n_jac_output_;
    e.v_int_type_ = v_int_type_;
    e.name_ = name.empty() ? name_ + "_copy" : name;
    e.q = e.in_args_[arg_idx(q)];
    e.v = e.in_args_[arg_idx(v)];
    e.a = e.in_args_[arg_idx(a)];
    e.q->name() = name.empty() ? q->name() + "_copy" : name + "_q";
    e.v->name() = name.empty() ? v->name() + "_copy" : name + "_v";
    e.a->name() = name.empty() ? a->name() + "_copy" : name + "_a";
    e.dt = dt; // share the same dt
    e.q->next()->name() = e.q->get_next_name();
    e.v->next()->name() = e.v->get_next_name();
    return func(std::move(e));
}
} // namespace multibody
} // namespace moto