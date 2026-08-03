#include <moto/core/linear_backend.hpp>
#include <moto/ocp/impl/node_data.hpp>
#include <moto/solver/ns_riccati/ns_riccati_data.hpp>
#include <moto/solver/ns_riccati/generic_solver.hpp>
#include <moto/ocp/dynamics.hpp>

namespace moto {
namespace solver {
namespace ns_riccati {
void ns_riccati_data::update_projected_dynamics() {
    full_data_->for_each(__dyn, [](const generic_dynamics &dyn, func_approx_data &data) {
        dyn.compute_project_jacobians(data);
    });
}
void ns_riccati_data::update_projected_dynamics_residual() {
    full_data_->for_each(__dyn, [](const generic_dynamics &dyn, func_approx_data &data) {
        dyn.compute_project_residual(data);
    });
}
void ns_riccati_data::apply_jac_y_inverse_transpose(vector &v, vector &dst) {
    dst.setZero();
    full_data_->for_each(__dyn, [&v, &dst](const generic_dynamics &dyn, func_approx_data &data) {
        const size_t start = data.problem()->get_expr_start(dyn);
        auto local_v = v.segment(start, dyn.dim());
        auto local_dst = dst.segment(start, dyn.dim());
        dyn.apply_jac_y_inverse_transpose(data, local_v, local_dst);
    });
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
  product(F_x, product_op::right_times, -1., ny, ny, ny, nx);
  product(F_x, product_op::right_transpose_times, -1., ny, 1, 1, nx);
  product(F_x, product_op::right_transpose_times, -1., ny, nx, nx, nx);
  product(F_u, product_op::times, -1., nu, 1, ny, 1);
  product(F_u, product_op::times, -1., nu, nx, ny, nx);
  product(F_u, product_op::transpose_times, -1., ny, 1, nu, 1);
  product(F_u, product_op::transpose_times, -1., ny, nx, nu, nx);
  prepare_weighted_gram(F_u);

  if (ns) {
    product(s_y, product_op::times, -1., ny, 1, ns, 1);
    product(s_y, product_op::transpose_times, -1., ns, 1, ny, 1);
    prepare_sparse_product(s_y, F_u, product_op::times, -1., ns, nu);
    prepare_sparse_product(s_y, F_x, product_op::times, -1., ns, nx);
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
  prepare_sparse_product(F_x, Q_yx, product_op::transpose_times, -1., nx, nx);
  prepare_sparse_product(F_x, Q_yx_mod, product_op::transpose_times, -1., nx,
                         nx);

  for (auto f : primal_fields)
    for (auto constr : constr_fields) {
      const auto &jac = dense_->approx_[constr].jac_[f];
      if (!jac.is_empty() && dense_->dual_[constr].size())
        product(jac, product_op::right_transpose_times, 1., jac.rows(), 1, 1,
                jac.cols());
    }
  prepare_products(products);
}

ns_riccati_data generic_solver::create_data(node_data *full_data) {
    return ns_riccati_data(full_data);
}

ns_riccati_data::ns_riccati_data(node_data *full_data)
    : solver::data_base(&full_data->sym_val(), &full_data->dense()),
      full_data_(full_data),
      ns(dense_->approx_[__eq_x].v_.size()),
      nc(dense_->approx_[__eq_xu].v_.size()), ncstr(ns + nc), d_u(nu, nx),
      d_y(nx, nx), d_lbd_f(nx), d_lbd_s_c_pre_solve(nu), d_lbd_s_c(ncstr),
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
    // nsp_->F_0_k.resize(nx);
    // nsp_->F_0_K.resize(nx, nx);
    // nsp_->F_u.resize(nx, nu);
    nsp_.Q_zz.resize(nu, nu);
    nsp_.s_0_p_k.resize(ns);
    nsp_.s_0_p_K.resize(ns, nx);
    nsp_.y_y_k.resize(nx);
    nsp_.y_y_K.resize(nx, nx);
    nsp_.u_y_k.resize(nu);
    nsp_.u_y_K.resize(nu, nx);
    nsp_.u_0_p_k.resize(nu);
    nsp_.u_0_p_K.resize(nu, nx);
    nsp_.y_0_p_k.resize(nx);
    nsp_.y_0_p_K.resize(nx, nx);
    nsp_.s_u.resize(ns, nu);
    nsp_.s_c_stacked.resize(ncstr, nu);
    nsp_.s_c_stacked_0_k.resize(ncstr);
    nsp_.s_c_stacked_0_K.resize(ncstr, nx);
    d_y.K.setZero();
    // if (nsp_->sparse_factorizer_)
    // nsp_->sparse_factorizer_->init(nsp_);
}

} // namespace ns_riccati
} // namespace solver
} // namespace moto
