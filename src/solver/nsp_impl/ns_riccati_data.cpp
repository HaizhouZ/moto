#include <moto/core/linear_backend.hpp>
#include <moto/ocp/impl/node_data.hpp>
#include <moto/solver/ns_riccati/ns_riccati_data.hpp>
#include <moto/solver/ns_riccati/generic_solver.hpp>
#include <moto/ocp/dynamics.hpp>
#include <moto/ocp/lifted.hpp>

#include <casadi/casadi.hpp>

#include <deque>
#include <unordered_map>

namespace moto {
namespace solver {
namespace ns_riccati {
namespace {

casadi::Sparsity casadi_sparsity(const linear_backend::matrix_layout &layout) {
    const auto pattern = linear_backend::analyze_pattern(layout);
    std::vector<casadi_int> rows;
    std::vector<casadi_int> cols;
    rows.reserve(pattern.nnz());
    cols.reserve(pattern.nnz());
    for (size_t col = 0; col < pattern.cols; ++col)
        for (size_t nz = pattern.colind[col]; nz < pattern.colind[col + 1]; ++nz) {
            rows.push_back(static_cast<casadi_int>(pattern.row[nz]));
            cols.push_back(static_cast<casadi_int>(col));
        }
    return casadi::Sparsity::triplet(
        static_cast<casadi_int>(pattern.rows),
        static_cast<casadi_int>(pattern.cols), rows, cols);
}

} // namespace

struct nsp_linear_plan {
    std::vector<sparse_matrix> workspace;
    linear_backend::graph_kernel presolve;
    std::vector<const sparse_matrix *> sparse_inputs;
    std::vector<scalar_t *> pointers;
    vector action_rhs, action_output;
    size_t dynamic_output = 0;
    bool integrated = false;
    bool projection_active = false;
};

namespace {

std::vector<const sparse_matrix *> presolve_sparse_inputs(
    ns_riccati_data &d) {
    return {
        &d.Q_xx,
        &d.Q_xx_mod,
        &d.Q_uu,
        &d.Q_uu_mod,
        &d.Q_ux,
        &d.Q_ux_mod,
        &d.Q_yx,
        &d.Q_yx_mod,
        &d.Q_lx,
        &d.Q_lx_mod,
        &d.dense_->lag_hess_[__y][__u],
        &d.dense_->hessian_modification_[__y][__u],
        &d.dense_->lag_hess_[__l][__u],
        &d.dense_->hessian_modification_[__l][__u],
        &d.dense_->lag_hess_[__l][__y],
        &d.dense_->hessian_modification_[__l][__y],
        &d.Q_ll,
        &d.Q_ll_mod,
    };
}

void append_writable_pointers(std::vector<scalar_t *> &destination,
                              const sparse_matrix &source) {
    auto pointers = linear_backend::panel_pointers(source);
    const size_t diagonal = source.diagonal_segments_.empty()
                                ? source.diag_panels_.size()
                                : source.diagonal_segments_.size();
    pointers.resize(source.dense_panels_.size() + diagonal);
    destination.insert(destination.end(), pointers.begin(), pointers.end());
}

void append_lifted_inputs(ns_riccati_data &d,
                          std::vector<scalar_t *> &pointers) {
    const auto &program =
        *d.dense_->prob_->linear_profile().lifted_program;
    for (const auto &binding : program.input_bindings) {
        switch (binding.source) {
        case lifted_graph_input_binding::kind::jacobian_panel: {
            auto input = linear_backend::panel_pointers(
                d.dense_->approx_[binding.equation].jac_[binding.variable]);
            if (binding.panels.empty()) {
                pointers.insert(pointers.end(), input.begin(), input.end());
            } else {
                for (const size_t panel : binding.panels)
                    pointers.push_back(input.at(panel));
            }
            break;
        }
        case lifted_graph_input_binding::kind::residual:
            pointers.push_back(
                d.dense_->approx_[binding.equation].v_.data());
            break;
        case lifted_graph_input_binding::kind::parameter:
            pointers.push_back((*d.lifting_.data)[*binding.parameter].data());
            break;
        }
    }
}

void append_lifted_outputs(ns_riccati_data &d,
                           std::vector<scalar_t *> &pointers) {
    append_writable_pointers(pointers, d.F_x);
    append_writable_pointers(pointers, d.lifting_.l_x());
    append_writable_pointers(pointers, d.F_u);
    append_writable_pointers(pointers, d.lifting_.l_u());
    auto &local = d.lifting_.data->as<generic_dynamics::approx_data>();
    for (const auto &spec :
         d.dense_->prob_->linear_profile().lifted_intermediates)
        append_writable_pointers(
            pointers, local.lifted_intermediates_.at(spec.name));
    pointers.push_back(d.F_0.data());
    pointers.push_back(d.lifting_.l_0().data());
}

struct nsp_layout_signature {
    size_t nx = 0, nu = 0, ny = 0, nl = 0;
    std::string lifted_artifact_identity;
    std::vector<linear_backend::matrix_layout> inputs;
    bool operator==(const nsp_layout_signature &) const = default;
};

struct nsp_layout_signature_hash {
    size_t operator()(const nsp_layout_signature &value) const noexcept {
        size_t seed = value.nx;
        const auto combine = [&seed](size_t item) {
            seed ^= item + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
        };
        combine(value.nu);
        combine(value.ny);
        combine(value.nl);
        for (const unsigned char c : value.lifted_artifact_identity)
            combine(c);
        for (const auto &layout : value.inputs) {
            combine(layout.rows);
            combine(layout.cols);
            for (const auto &panel : layout.panels) {
                combine(static_cast<size_t>(panel.pattern));
                combine(panel.row_offset);
                combine(panel.col_offset);
                combine(panel.rows);
                combine(panel.cols);
                combine(panel.transposed);
                combine(panel.storage_offset);
                combine(panel.storage_rows);
            }
        }
        return seed;
    }
};

nsp_layout_signature presolve_layout_signature(ns_riccati_data &d) {
    nsp_layout_signature result{d.nx, d.nu, d.ny, d.nl};
    if (const auto &program =
            d.dense_->prob_->linear_profile().lifted_program) {
        result.lifted_artifact_identity = program->artifact_identity;
        result.inputs.insert(result.inputs.end(), program->input_layouts.begin(),
                             program->input_layouts.end());
    }
    for (const auto *input : presolve_sparse_inputs(d))
        result.inputs.push_back(linear_backend::describe(*input));
    return result;
}

linear_backend::graph_kernel build_unconstrained_presolve_graph(
    ns_riccati_data &d) {
    using namespace linear_backend;
    std::vector<casadi::MX> inputs;
    std::vector<matrix_layout> layouts;
    const auto &lifted_program =
        d.dense_->prob_->linear_profile().lifted_program;
    if (lifted_program) {
        inputs = lifted_program->inputs;
        layouts = lifted_program->input_layouts;
        inputs.push_back(lifted_program->action_rhs);
        layouts.emplace_back();
        inputs.push_back(lifted_program->transpose_rhs);
        layouts.emplace_back();
    }
    const auto sparse_inputs = presolve_sparse_inputs(d);
    size_t sparse_index = 0;
    const auto sparse_input = [&](const char *name) {
        const auto layout = describe(*sparse_inputs[sparse_index++]);
        inputs.push_back(casadi::MX::sym(name, casadi_sparsity(layout)));
        layouts.push_back(layout);
        return inputs.back();
    };
    const auto q_xx = sparse_input("q_xx");
    const auto q_xx_mod = sparse_input("q_xx_mod");
    const auto q_uu = sparse_input("q_uu");
    const auto q_uu_mod = sparse_input("q_uu_mod");
    const auto q_ux = sparse_input("q_ux");
    const auto q_ux_mod = sparse_input("q_ux_mod");
    const auto q_yx = sparse_input("q_yx");
    const auto q_yx_mod = sparse_input("q_yx_mod");
    const auto q_lx = sparse_input("q_lx");
    const auto q_lx_mod = sparse_input("q_lx_mod");
    const auto h_yu = sparse_input("h_yu");
    const auto h_yu_mod = sparse_input("h_yu_mod");
    const auto h_lu = sparse_input("h_lu");
    const auto h_lu_mod = sparse_input("h_lu_mod");
    const auto h_ly = sparse_input("h_ly");
    const auto h_ly_mod = sparse_input("h_ly_mod");
    const auto q_ll = sparse_input("q_ll");
    const auto q_ll_mod = sparse_input("q_ll_mod");
    casadi::MX z_y, z_l, y_y, l_y;
    if (lifted_program) {
        const auto block_rows = [&](const casadi::MX &value, field_t field,
                                    size_t offset) {
            std::vector<casadi::MX> blocks;
            for (const sym &variable : d.dense_->prob_->exprs(field)) {
                const size_t rows = variable.tdim();
                blocks.push_back(value(
                    casadi::Slice(static_cast<casadi_int>(offset),
                                  static_cast<casadi_int>(offset + rows)),
                    casadi::Slice()));
                offset += rows;
            }
            return casadi::MX::vertcat(blocks);
        };
        y_y = block_rows(lifted_program->response_x, __y, 0);
        l_y = block_rows(lifted_program->response_x, __l, d.ny);
        z_y = -block_rows(lifted_program->response_u, __y, 0);
        z_l = -block_rows(lifted_program->response_u, __l, d.ny);
    } else {
        z_y = casadi::MX::sym("z_y", d.ny, d.nu);
        z_l = casadi::MX::sym("z_l", d.nl, d.nu);
        y_y = casadi::MX::sym("y_y", d.ny, d.nx);
        l_y = casadi::MX::sym("l_y", d.nl, d.nx);
        inputs.insert(inputs.end(), {z_y, z_l, y_y, l_y});
        layouts.resize(inputs.size());
    }

    const auto product = [](const casadi::MX &lhs, const casadi::MX &rhs) {
        return casadi::MX::mtimes(lhs, rhs);
    };
    const auto h_yu_total = h_yu + h_yu_mod;
    const auto h_lu_total = h_lu + h_lu_mod;
    const auto h_ly_total = h_ly + h_ly_mod;
    const auto q_ll_total = q_ll + q_ll_mod;
    const auto basis_u = q_uu + q_uu_mod +
                         product(h_yu_total.T(), z_y) +
                         product(h_lu_total.T(), z_l);
    const auto basis_y = h_yu_total + product(h_ly_total.T(), z_l);
    const auto basis_l = h_lu_total + product(h_ly_total, z_y) +
                         product(q_ll_total, z_l);
    const auto qzz = basis_u + product(z_y.T(), basis_y) +
                     product(z_l.T(), basis_l);

    const auto state_u = q_ux + q_ux_mod -
                         product(h_yu_total.T(), y_y) -
                         product(h_lu_total.T(), l_y);
    const auto state_y = q_yx + q_yx_mod -
                         product(h_ly_total.T(), l_y);
    const auto state_l = q_lx + q_lx_mod - product(h_ly_total, y_y) -
                         product(q_ll_total, l_y);
    const auto z_0_K = state_u + product(z_y.T(), state_y) +
                       product(z_l.T(), state_l);
    const auto v_xx = q_xx + q_xx_mod -
                      product((q_yx + q_yx_mod).T(), y_y) -
                      product((q_lx + q_lx_mod).T(), l_y) -
                      product(y_y.T(), state_y) -
                      product(l_y.T(), state_l);
    std::vector<casadi::MX> presolve_outputs;
    std::vector<std::vector<casadi::MX>> entries;
    if (lifted_program) {
        presolve_outputs = lifted_program->projection_outputs;
        presolve_outputs.insert(
            presolve_outputs.end(),
            {casadi::MX::densify(z_y), casadi::MX::densify(y_y),
             casadi::MX::densify(qzz), casadi::MX::densify(z_0_K),
             casadi::MX::densify(v_xx)});
        entries = {
            std::move(presolve_outputs),
            {lifted_program->action_output},
            {lifted_program->transpose_output}};
    } else {
        entries = {{casadi::MX::densify(qzz),
                    casadi::MX::densify(z_0_K),
                    casadi::MX::densify(v_xx)}};
    }
    const std::string artifact_identity =
        lifted_program
            ? "nsp_integrated_presolve_v3_" +
                  lifted_program->artifact_identity
            : "nsp_unconstrained_presolve_v2";
    return compile_graph(artifact_identity,
                         inputs, entries, layouts, nullptr,
                         "gen/linear_backend",
                         lifted_program
                             ? std::span<const casadi::MX>(
                                   lifted_program->spd_factors)
                             : std::span<const casadi::MX>{});
}

void attach_unconstrained_presolve_graph(
    ns_riccati_data &d, const linear_backend::graph_kernel &program) {
    d.linear_plan_ = std::make_shared<nsp_linear_plan>();
    auto &plan = *d.linear_plan_;
    plan.sparse_inputs = presolve_sparse_inputs(d);
    plan.presolve = program.instantiate(&plan.workspace);
    plan.integrated = static_cast<bool>(
        d.dense_->prob_->linear_profile().lifted_program);
    if (plan.integrated) {
        append_lifted_inputs(d, plan.pointers);
        plan.action_rhs.setZero(d.ny + d.nl);
        plan.action_output.setZero(d.ny + d.nl);
        plan.pointers.push_back(plan.action_rhs.data());
        plan.pointers.push_back(plan.action_rhs.data());
    }
    for (const auto *input : plan.sparse_inputs) {
        auto pointers = linear_backend::panel_pointers(*input);
        plan.pointers.insert(plan.pointers.end(), pointers.begin(),
                             pointers.end());
    }
    if (plan.pointers.size() != plan.presolve.input_count())
        throw std::logic_error(fmt::format(
            "NSP presolve graph input layout mismatch: bound={} graph={}",
            plan.pointers.size(), plan.presolve.input_count()));
    if (!plan.integrated) {
        plan.pointers.resize(plan.pointers.size() + 4);
    } else {
        append_lifted_outputs(d, plan.pointers);
    }
    plan.dynamic_output = plan.pointers.size();
    plan.pointers.resize(plan.pointers.size() + (plan.integrated ? 5 : 3));
    if (plan.integrated) {
        plan.pointers.push_back(plan.action_output.data());
        plan.pointers.push_back(plan.action_output.data());
    }
    if (plan.pointers.size() != plan.presolve.pointer_count() ||
        plan.presolve.entry_count() != (plan.integrated ? 3 : 1))
        throw std::logic_error("NSP presolve graph pointer layout mismatch");
}

} // namespace

void ns_riccati_data::update_projected_dynamics() {
    full_data_->for_each(__dyn, [](const generic_dynamics &group, func_approx_data &data) {
        group.compute_project_jacobians(data);
    });
}
void ns_riccati_data::update_projected_dynamics_residual() {
    full_data_->for_each(__dyn, [](const generic_dynamics &group, func_approx_data &data) {
        group.compute_project_residual(data);
    });
}
void ns_riccati_data::apply_jac_y_inverse_transpose(vector_ref v, vector &dst) {
    if (linear_plan_ && linear_plan_->projection_active) {
        lifting_.dual_rhs.setZero(ny + nl, 1);
        lifting_.dual_rhs.topRows(ny) = v;
        solve_integrated_lifted_system(
            lifting_.dual_rhs, lifting_.dual, true);
        dst = lifting_.dual.topRows(ny);
        return;
    }
    dst.setZero();
    full_data_->for_each(__dyn, [&v, &dst](const generic_dynamics &group, func_approx_data &data) {
        const size_t start = data.problem()->get_expr_start(group);
        auto local_v = v.segment(start, group.dim());
        auto local_dst = dst.segment(start, group.dim());
        group.apply_lifted_jacobian_inverse_transpose(data, local_v,
                                                       local_dst);
    });
}

void ns_riccati_data::prepare_lifting_operator() {
    const size_t nl = dense_->prob_->tdim(__l);
    const size_t ny_local = dense_->prob_->tdim(__y);
    const size_t lift_rows = dense_->prob_->dim(__lift);
    if (nl &&
        (dense_->prob_->dim(__dyn) != ny_local || lift_rows != nl))
        throw std::runtime_error(fmt::format(
            "stage lifting operator requires dim(dyn)=tdim(y) and "
            "dim(lift)=tdim(l): dyn={}, y={}, lift={}, l={}",
            dense_->prob_->dim(__dyn), ny_local, lift_rows, nl));

    auto &op = lifting_;
    op.ny = ny_local;
    op.nl = nl;
    full_data_->for_each(__dyn, [&](const generic_dynamics &group,
                                    func_approx_data &data) {
        if (!group.owns_stage_elimination()) return;
        if (op.function)
            throw std::runtime_error(
                "stage has multiple lifted elimination owners");
        auto &local = data.as<generic_dynamics::approx_data>();
        op.function = &group;
        op.data = &data;
        op.projected_l_x = &local.proj_l_x_;
        op.projected_l_u = &local.proj_l_u_;
        op.projected_l_0 = &local.proj_l_res_;
    });
    if (nl && (!op.function || !op.data))
        throw std::runtime_error(
            "stage with explicit lifted variables requires one "
            "generic_lifted elimination owner");

    if (std::getenv("MOTO_DEBUG_LIFTING") && nl) {
        fmt::println(
            "lifting stage: system={}x{} |Px|={} |Pu|={} |p0|={}",
            ny_local + nl, ny_local + nl,
            std::max(F_x.dense().cwiseAbs().maxCoeff(),
                     op.l_x().dense().cwiseAbs().maxCoeff()),
            std::max(F_u.dense().cwiseAbs().maxCoeff(),
                     op.l_u().dense().cwiseAbs().maxCoeff()),
            std::max(F_0.cwiseAbs().maxCoeff(),
                     op.l_0().cwiseAbs().maxCoeff()));
    }
}

void ns_riccati_data::update_lifted_basis_K() {
    if (rank_status_ == rank_status::unconstrained) {
        nsp_.Z_y.setZero(ny, nu);
        nsp_.Z_l.setZero(nl, nu);
        linear_backend::write_dense(F_u, nsp_.Z_y, {.alpha = -1.});
        linear_backend::write_dense(lifting_.l_u(), nsp_.Z_l,
                                    {.alpha = -1.});
    } else {
        nsp_.Z_y.setZero();
        nsp_.Z_l.setZero();
        linear_backend::multiply(F_u, nsp_.Z_u, nsp_.Z_y, -1.);
        linear_backend::multiply(lifting_.l_u(), nsp_.Z_u, nsp_.Z_l, -1.);
    }
    nsp_.y_y_K.setZero();
    nsp_.l_y_K.setZero();
    linear_backend::write_dense(F_x, nsp_.y_y_K);
    linear_backend::write_dense(lifting_.l_x(), nsp_.l_y_K);
    linear_backend::multiply(F_u, nsp_.u_y_K, nsp_.y_y_K, -1.);
    linear_backend::multiply(lifting_.l_u(), nsp_.u_y_K, nsp_.l_y_K, -1.);
}

void ns_riccati_data::update_lifted_basis_k() {
    nsp_.y_y_k = F_0;
    nsp_.l_y_k = lifting_.l_0();
    linear_backend::multiply(F_u, nsp_.u_y_k, nsp_.y_y_k, -1.);
    linear_backend::multiply(lifting_.l_u(), nsp_.u_y_k, nsp_.l_y_k, -1.);
}

void ns_riccati_data::build_lifted_hard_geometry(
    matrix *C_u, matrix *C_x, vector *c_0) {
    const auto &approx = dense_->approx_;
    const size_t rows = ns + nc;
    if (C_u) C_u->setZero(rows, nu);
    if (C_x) C_x->setZero(rows, nx);
    if (c_0) c_0->setZero(rows);
    size_t row = 0;
    for (const auto cf : hard_constr_fields_non_dyn) {
        const size_t n = approx[cf].v_.size();
        if (!n) continue;
        if (C_u) {
            auto dst = C_u->middleRows(row, n);
            dst.setZero();
            linear_backend::write_dense(approx[cf].jac_[__u], dst);
            linear_backend::right_multiply(approx[cf].jac_[__y], F_u, dst,
                                           -1.);
            linear_backend::right_multiply(approx[cf].jac_[__l],
                                           lifting_.l_u(), dst, -1.);
        }
        if (C_x) {
            auto dst = C_x->middleRows(row, n);
            dst.setZero();
            linear_backend::write_dense(approx[cf].jac_[__x], dst);
            linear_backend::right_multiply(approx[cf].jac_[__y], F_x, dst,
                                           -1.);
            linear_backend::right_multiply(approx[cf].jac_[__l],
                                           lifting_.l_x(), dst, -1.);
        }
        if (c_0) {
            auto dst = c_0->segment(row, n);
            dst = approx[cf].v_;
            linear_backend::multiply(approx[cf].jac_[__y], F_0, dst, -1.);
            linear_backend::multiply(approx[cf].jac_[__l], lifting_.l_0(),
                                     dst, -1.);
        }
        row += n;
    }
}

void ns_riccati_data::recover_lifted_dual(vector_ref projected_y_rhs) {
    if (!lifting_.function) {
        apply_jac_y_inverse_transpose(projected_y_rhs,
                                      trial_dual_step[__dyn]);
        return;
    }
    lifting_.dual_rhs.resize(lifting_.ny + lifting_.nl, 1);
    lifting_.dual_rhs.topRows(lifting_.ny) = projected_y_rhs;
    lifting_.dual_rhs.bottomRows(lifting_.nl) = -Q_l.transpose();
    auto lifted_rhs = lifting_.dual_rhs.bottomRows(lifting_.nl);
    for (const auto field : primal_fields) {
        const auto row = std::max(__l, field);
        const auto col = std::min(__l, field);
        linear_backend::multiply(
            dense_->lag_hess_[row][col], trial_prim_step[field],
            lifted_rhs, -1.);
        linear_backend::multiply(
            dense_->hessian_modification_[row][col],
            trial_prim_step[field], lifted_rhs, -1.);
    }
    if (linear_plan_ && linear_plan_->projection_active)
        solve_integrated_lifted_system(
            lifting_.dual_rhs, lifting_.dual, true);
    else
        lifting_.function->solve_stage_lifted_system(
            *lifting_.data, lifting_.dual_rhs, lifting_.dual, true);
    trial_dual_step[__dyn] = lifting_.dual.topRows(lifting_.ny);
    trial_dual_step[__lift] = lifting_.dual.bottomRows(lifting_.nl);
    if (std::getenv("MOTO_DEBUG_LIFTING_DUAL") && lifting_.nl <= 4) {
        fmt::println("lifted dual rhs={} solved={} ql={} du={}",
                     lifting_.dual_rhs.transpose(), lifting_.dual.transpose(),
                     Q_l, trial_prim_step[__u].transpose());
    }
}

void ns_riccati_data::prepare_linear_backend() {
  using namespace linear_backend;
  std::vector<product_request> products;
  const auto product = [&](const sparse_matrix &a, product_op op, scalar_t sign,
                           size_t br, size_t bc, size_t cr, size_t cc) {
    products.push_back({&a, op, sign, br, bc, cr, cc});
  };
  const auto dump = [](const sparse_matrix &a, size_t rows, scalar_t alpha = 1.,
                       bool overwrite = false) {
    prepare_dense_write(a, rows, alpha, overwrite);
  };

  for (auto *q : {&Q_xx, &Q_uu, &Q_yy}) {
    dump(*q, q->rows());
    product(*q, product_op::times, 1., q->cols(), 1, q->rows(), 1);
  }
  for (auto *q : {&Q_xx_mod, &Q_uu_mod, &Q_yy_mod})
    dump(*q, q->rows());
  product(Q_yx_mod, product_op::times, -1., nx, 1, ny, 1);
  product(Q_ux_mod, product_op::times, -1., nx, 1, nu, 1);
  product(Q_uu, product_op::times, -1., nu, 1, nu, 1);
  product(Q_uu_mod, product_op::times, -1., nu, 1, nu, 1);
  for (auto *q : {&Q_ux, &Q_ux_mod}) {
    dump(*q, q->rows());
    product(*q, product_op::right_transpose_times, -1., q->rows(), 1, 1,
            q->cols());
    product(*q, product_op::right_transpose_times, -1., q->rows(), nx, nx,
            q->cols());
  }
  for (auto *q : {&Q_yx, &Q_yx_mod}) {
    dump(*q, q->rows());
    product(*q, product_op::right_transpose_times, -1., q->rows(), 1, 1,
            q->cols());
    product(*q, product_op::right_transpose_times, -1., q->rows(), nx, nx,
            q->cols());
  }

  dump(F_x, F_x.rows());
  dump(F_x, F_x.rows(), -1.);
  product(F_u, product_op::times, -1., nu, 1, ny, 1);
  product(F_u, product_op::times, -1., nu, nx, ny, nx);
  product(F_u, product_op::transpose_times, -1., ny, 1, nu, 1);
  product(F_u, product_op::transpose_times, -1., ny, nx, nu, nx);

  if (const size_t lifted_dim = dense_->prob_->tdim(__l)) {
    dump(Q_ll, lifted_dim);
    product(Q_ll, product_op::times, 1., lifted_dim, 1, lifted_dim, 1);
    dump(Q_ll_mod, lifted_dim);
    dump(lifting_.l_x(), lifted_dim);
    for (size_t cols : {size_t(1), nu, nz, nx}) {
      if (!cols) continue;
      product(lifting_.l_u(), product_op::times, -1., nu, cols,
              lifted_dim, cols);
    }
    constexpr auto fields = std::array{__u, __y, __l};
    const std::array<size_t, 3> dim{nu, ny, lifted_dim};
    for (size_t row = 0; row < 3; ++row) {
      for (const auto *Q : {&dense_->lag_hess_[fields[row]][__x],
                            &dense_->hessian_modification_[fields[row]][__x]}) {
        dump(*Q, dim[row]);
        product(*Q, product_op::transpose_times, -1., dim[row], 1, nx, 1);
      }
      for (size_t col = 0; col < 3; ++col) {
        if (row == 1 && col == 1) continue;
        const auto op = row >= col ? product_op::times
                                   : product_op::transpose_times;
        const auto hi = std::max(fields[row], fields[col]);
        const auto lo = std::min(fields[row], fields[col]);
        for (const auto *Q : {&dense_->lag_hess_[hi][lo],
                              &dense_->hessian_modification_[hi][lo]}) {
          for (size_t cols : {size_t(1), nu, nz, nx}) {
            if (!cols) continue;
            product(*Q, op, 1., dim[col], cols, dim[row], cols);
          }
        }
      }
    }
  }

  for (const auto field : hard_constr_fields_non_dyn) {
    const size_t rows = dense_->approx_[field].v_.size();
    if (!rows) continue;
    const auto &jac_y = dense_->approx_[field].jac_[__y];
    prepare_sparse_product(F_u, jac_y, product_op::right_times, -1., rows,
                           nu);
    prepare_sparse_product(F_x, jac_y, product_op::right_times, -1., rows,
                           nx);
    if (dense_->prob_->tdim(__l)) {
      const auto &jac_l = dense_->approx_[field].jac_[__l];
      prepare_sparse_product(lifting_.l_u(), jac_l,
                             product_op::right_times, -1., rows, nu);
      prepare_sparse_product(lifting_.l_x(), jac_l,
                             product_op::right_times, -1., rows, nx);
    }
  }

  if (ns) {
    product(s_y, product_op::times, -1., ny, 1, ns, 1);
    product(s_y, product_op::transpose_times, -1., ns, 1, ny, 1);
    dump(s_x, ns);
  }
  if (nc) {
    dump(c_u, nc, 1., true);
    dump(c_x, nc);
  }

  for (size_t cols : {nu, nz, nx}) {
    if (!cols)
      continue;
    product(Q_uu, product_op::times, 1., nu, cols, nu, cols);
    product(Q_uu, product_op::times, -1., nu, cols, nu, cols);
    product(Q_uu_mod, product_op::times, 1., nu, cols, nu, cols);
    product(Q_uu_mod, product_op::times, -1., nu, cols, nu, cols);
    product(F_u, product_op::times, -1., nu, cols, ny, cols);
  }
  for (auto f : primal_fields)
    for (auto constr : constr_fields) {
      const auto &jac = dense_->approx_[constr].jac_[f];
      if (!jac.is_empty() && dense_->dual_[constr].size())
        product(jac, product_op::right_transpose_times, 1., jac.rows(), 1, 1,
                jac.cols());
  }
  prepare_products(products);
}

void generic_solver::prepare_ocp_linear_graph(
    std::span<ns_riccati_data *> stages) {
    std::unordered_map<nsp_layout_signature,
                       const linear_backend::graph_kernel *,
                       nsp_layout_signature_hash>
        programs;
    std::deque<linear_backend::graph_kernel> owned_programs;

    for (auto *stage : stages) {
        if (!stage || !stage->nl) continue;
        const auto key = presolve_layout_signature(*stage);
        if (stage->has_unconstrained_presolve_graph()) {
            programs.try_emplace(key, &stage->linear_plan_->presolve);
            continue;
        }
        auto found = programs.find(key);
        if (found == programs.end()) {
            owned_programs.push_back(
                build_unconstrained_presolve_graph(*stage));
            found = programs.emplace(key, &owned_programs.back()).first;
        }
        attach_unconstrained_presolve_graph(*stage, *found->second);
    }
}

bool ns_riccati_data::has_unconstrained_presolve_graph() const {
    return linear_plan_ && linear_plan_->presolve;
}

bool ns_riccati_data::has_integrated_presolve_graph() const {
    return linear_plan_ && linear_plan_->integrated;
}

bool ns_riccati_data::uses_sparse_lifted_basis() const {
    return has_integrated_presolve_graph() &&
           rank_status_ == rank_status::unconstrained;
}

void ns_riccati_data::run_unconstrained_presolve_graph() {
    auto &plan = *linear_plan_;
    size_t offset = plan.dynamic_output;
    if (!plan.integrated) {
        plan.pointers[offset - 4] = nsp_.Z_y.data();
        plan.pointers[offset - 3] = nsp_.Z_l.data();
        plan.pointers[offset - 2] = nsp_.y_y_K.data();
        plan.pointers[offset - 1] = nsp_.l_y_K.data();
        plan.pointers[offset++] = nsp_.Q_zz.data();
        plan.pointers[offset++] = nsp_.z_0_K.data();
        plan.pointers[offset] = V_xx.data();
    } else {
        plan.pointers[offset++] = nsp_.Z_y.data();
        plan.pointers[offset++] = nsp_.y_y_K.data();
        plan.pointers[offset++] = nsp_.Q_zz.data();
        plan.pointers[offset++] = nsp_.z_0_K.data();
        plan.pointers[offset] = V_xx.data();
    }
    plan.presolve(0, plan.pointers);
    plan.projection_active = plan.integrated;
}

void ns_riccati_data::solve_integrated_lifted_system(
    const matrix &rhs, matrix &destination, bool transpose) {
    auto &plan = *linear_plan_;
    if (!plan.integrated || !plan.projection_active)
        throw std::logic_error("integrated lifted factorization is not active");
    if (rhs.rows() != plan.action_rhs.size())
        throw std::invalid_argument(
            "integrated lifted action right-hand side row mismatch");
    destination.resize(rhs.rows(), rhs.cols());
    for (Eigen::Index column = 0; column < rhs.cols(); ++column) {
        plan.action_rhs = rhs.col(column);
        plan.presolve(transpose ? 2 : 1, plan.pointers);
        destination.col(column) = plan.action_output;
    }
}

ns_riccati_data generic_solver::create_data(node_data *full_data) {
    return ns_riccati_data(full_data);
}

ns_riccati_data::ns_riccati_data(node_data *full_data)
    : solver::data_base(&full_data->sym_val(), &full_data->dense()),
      full_data_(full_data),
      ns(dense_->approx_[__eq_x].v_.size()),
      nc(dense_->approx_[__eq_xu].v_.size()), ncstr(ns + nc), d_u(nu, nx),
      d_y(nx, nx), d_lbd_f(nx),
      d_lbd_s_c_pre_solve(nu), d_lbd_s_c(ncstr),
      F_x(dense_->proj_f_x()),
      F_u(dense_->proj_f_u()),
      s_y(dense_->approx_[__eq_x].jac_[__y]),
      s_x(dense_->approx_[__eq_x].jac_[__x]),
      c_x(dense_->approx_[__eq_xu].jac_[__x]),
      c_u(dense_->approx_[__eq_xu].jac_[__u]),
      F_0(dense_->proj_f_res()) {
    if (nu < ncstr) {
        nz = 0;
        // throw std::runtime_error("system over-constrained, i.e., nu < ncstr");
    } else {
        nz = nu - ncstr;
    }
    nsp_.Q_zz.resize(nu, nu);
    nsp_.s_0_p_k.resize(ns);
    nsp_.s_0_p_K.resize(ns, nx);
    nsp_.y_y_k.resize(nx);
    nsp_.y_y_K.resize(nx, nx);
    nsp_.Z_l.resize(dense_->prob_->tdim(__l), nu);
    nsp_.l_y_k.resize(dense_->prob_->tdim(__l));
    nsp_.l_y_K.resize(dense_->prob_->tdim(__l), nx);
    nsp_.u_y_k.resize(nu);
    nsp_.u_y_K.resize(nu, nx);
    nsp_.u_0_p_k.resize(nu);
    nsp_.u_0_p_K.resize(nu, nx);
    nsp_.y_0_p_k.resize(nx);
    nsp_.y_0_p_K.resize(nx, nx);
    nsp_.l_0_p_k.resize(dense_->prob_->tdim(__l));
    nsp_.l_0_p_K.resize(dense_->prob_->tdim(__l), nx);
    nsp_.s_u.resize(ns, nu);
    nsp_.s_c_stacked.resize(ncstr, nu);
    nsp_.s_c_stacked_0_k.resize(ncstr);
    nsp_.s_c_stacked_0_K.resize(ncstr, nx);
    d_y.K.setZero();
    const size_t nl = dense_->prob_->tdim(__l);
    lifting_.empty_l_x.resize(nl, nx);
    lifting_.empty_l_u.resize(nl, nu);
    lifting_.empty_l_0.setZero(nl);
    lifting_.projected_l_x = &lifting_.empty_l_x;
    lifting_.projected_l_u = &lifting_.empty_l_u;
    lifting_.projected_l_0 = &lifting_.empty_l_0;
    prepare_lifting_operator();
}

} // namespace ns_riccati
} // namespace solver
} // namespace moto
