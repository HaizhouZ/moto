#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <moto/ocp/problem.hpp>
#include <moto/utils/codegen.hpp>
#define ENABLE_TIMED_BLOCK
#include <moto/multibody/stacked_euler.hpp>
#include <moto/utils/timed_block.hpp>

#include <Eigen/LU>
#include <moto/spmm/sparse_mat.hpp>

// TEST_CASE("multibody_euler") {
//     using namespace moto;
//     using dyn_t = multibody::stacked_euler;
//     auto dt = sym::inputs("dt", 1);
//     func dyn = dyn_t("stacked_euler");
//     auto e = multibody::euler::from_urdf("rsc/go2_description/urdf/go2_description.urdf", dt);
//     auto e2 = multibody::euler::from_urdf("rsc/iiwa_description/urdf/iiwa14.urdf", dt,
//                                           multibody::root_joint_t::fixed,
//                                           multibody::euler::v_int_type::_explicit);
//     auto e3 = multibody::euler::from_urdf("rsc/iiwa_description/urdf/iiwa14.urdf", dt,
//                                           multibody::root_joint_t::fixed,
//                                           multibody::euler::v_int_type::_mid_point);
//     dyn.as<dyn_t>().add(e);
//     dyn.as<dyn_t>().add(e2);
//     dyn.as<dyn_t>().add(e3);
//     dyn.as<dyn_t>().add(e.as<multibody::euler>().share());

//     auto prob = ocp::create();
//     prob->add(dyn);

//     prob->print_summary();

//     auto s_data = sym_data(prob.get());
//     auto m_data = merit_data(prob.get());
//     auto sh_data = shared_data(prob.get(), &s_data);

//     auto d_ptr = dyn->create_approx_data(s_data, m_data, sh_data);
//     auto &d = *d_ptr;

//     s_data.value_[__x].setRandom();
//     s_data.value_[__u].setRandom();
//     s_data.value_[__y].setRandom();
//     auto &dual = m_data.dual_[__dyn].setRandom();

//     dyn->compute_approx(d, true, true, false);
//     // dyn2->compute_approx(d2, true, true, false);

//     dyn.as<dyn_t>().compute_project_derivatives(d);

//     // fmt::println("f_x dense:\n{:.1}", m_data.approx_[__dyn].jac_[__x].dense());
//     // fmt::println("f_y dense:\n{:.1}", m_data.approx_[__dyn].jac_[__y].dense());
//     // fmt::println("f_u dense:\n{:.1}", m_data.approx_[__dyn].jac_[__u].dense());
//     // fmt::println("proj_f_x dense:\n{}", m_data.proj_f_x().dense());
//     // fmt::println("proj_f_u dense:\n{}", m_data.proj_f_u().dense());
//     auto lu_ = m_data.approx_[__dyn].jac_[__y].dense().lu();
//     {
//         vector ground_truth = lu_.solve(m_data.approx_[__dyn].v_);
//         REQUIRE(m_data.proj_f_res().isApprox(ground_truth));
//     }
//     {
//         matrix ground_truth = lu_.solve(m_data.approx_[__dyn].jac_[__x].dense());
//         // fmt::println("groudn_Truth:\n{}", ground_truth);
//         // fmt::println("residual:\n{}", m_data.proj_f_x().dense() - ground_truth);
//         REQUIRE(m_data.proj_f_x().dense().isApprox(ground_truth));
//     }
//     {
//         matrix ground_truth = lu_.solve(m_data.approx_[__dyn].jac_[__u].dense());
//         // fmt::println("groudn_Truth:\n{}", ground_truth);
//         // fmt::println("residual:\n{}", m_data.proj_f_u().dense() - ground_truth);
//         REQUIRE(m_data.proj_f_u().dense().isApprox(ground_truth));
//     }
// }
void __attribute__((noinline)) dense_inner_product(const moto::matrix &rhs, moto::matrix &D, moto::matrix &out) {
    out.noalias() = rhs.transpose() * D * rhs;
}
void __attribute__((noinline)) sp_inner_product(size_t nv, moto::matrix &cache, moto::matrix &U,
                                                moto::aligned_map_t &f_t, moto::aligned_map_t &f_u_v_diag,
                                                moto::matrix &hess) {
    cache.leftCols(nv).noalias() += hess.middleCols(nv, nv) * f_u_v_diag.asDiagonal();
    cache.rightCols(1).noalias() += hess.middleCols(0, 2 * nv) * f_t;
    U.topRows(nv).noalias() += f_u_v_diag.asDiagonal() * cache.bottomRows(nv);
    U.bottomRows(1).noalias() += f_t.transpose() * cache;
}
TEST_CASE("multibody_euler_speed") {
    using namespace moto;
    using dyn_t = multibody::stacked_euler;
    auto dt = sym::inputs("dt", 1);
    func dyn = dyn_t("stacked_euler");
    // auto e = multibody::euler::from_urdf("rsc/iiwa_description/urdf/iiwa14.urdf", dt, multibody::root_joint_t::fixed, multibody::euler::v_int_type::_explicit);
    auto e = multibody::euler::from_urdf("rsc/go2_description/urdf/go2_description.urdf", dt); //multibody::root_joint_t::xyz_eulerZYX, multibody::euler::v_int_type::_explicit);
    auto b = multibody::euler::from_urdf("rsc/go2_description/urdf/box_description.urdf", dt);//, multibody::root_joint_t::xyz_eulerZYX);//, multibody::euler::v_int_type::_explicit);

    dyn.as<dyn_t>().add(e);
    // dyn.as<dyn_t>().add(b);
    // dyn.as<dyn_t>().add(b.as<multibody::euler>().share());
    // dyn.as<dyn_t>().add(b.as<multibody::euler>().share());

    auto prob = ocp::create();
    prob->add(dt);
    prob->add(dyn);
    prob->print_summary();
    bool show = true;
    size_t N_trials = 100;
    while (N_trials--) {
        size_t n_trials = 100;

        std::vector<sym_data> s_data;
        s_data.reserve(n_trials);
        for (size_t i = n_trials; i--;)
            s_data.emplace_back(prob.get());
        std::vector<merit_data> m_data;
        m_data.reserve(n_trials);
        for (size_t i = n_trials; i--;)
            m_data.emplace_back(prob.get());
        std::vector<shared_data> sh_data;
        sh_data.reserve(n_trials);
        for (size_t i = n_trials; i--;)
            sh_data.emplace_back(prob.get(), &s_data[i]);
        std::vector<matrix> hess_;
        std::vector<multibody::stacked_euler::approx_data> ds_;
        size_t nv = dyn.as<multibody::stacked_euler>().nv_;
        std::vector<sparse_mat> Z_u;

        for (size_t i = n_trials; i--;) {
            auto d_ptr = dyn->create_approx_data(s_data[i], m_data[i], sh_data[i]);
            auto &d = *d_ptr;
            ds_.emplace_back(d.as<multibody::stacked_euler::approx_data>());
            // auto d2_ptr = dyn2->create_approx_data(s_data, m_data, sh_data);
            // auto &d2 = d2_ptr->as<euler::impl::approx_data>();

            s_data[i].value_[__x].setRandom();
            s_data[i].value_[__u].setRandom();
            s_data[i].value_[__y].setRandom();
            auto hess = matrix::Random(prob->tdim(__y), prob->tdim(__y));
            hess_.emplace_back(hess * hess.transpose());

            dyn->compute_approx(d, true, true, false);
            dyn.as<multibody::stacked_euler>().compute_project_derivatives(d);
            if (show) {
                show = false;
                // fmt::println("f_y dense:\n{:.1}", m_data[i].approx_[__dyn].jac_[__u].dense());
                fmt::println("f_y dense:\n{:.1}", m_data[i].proj_f_u().dense());
            }
        }
        thread_local matrix U;
        U.resize(prob->dim(__u), prob->dim(__u));
        thread_local matrix cache;
        cache.resize(prob->tdim(__y), prob->tdim(__y));
        thread_local matrix Q_zz;
        // Q_zz.resize(12, 12);
        Q_zz.resize(prob->tdim(__u), prob->tdim(__u));
        // matrix cache(prob->tdim(__y), prob->tdim(__y));
        U.setZero();

        // dyn2->compute_approx(d2, true, true, false);
        // dyn2->compute_project_derivatives(d2);
        std::vector<matrix> dense_f_u, dense_zu;
        for (size_t i = n_trials; i--;) {
            dense_f_u.push_back(m_data[i].proj_f_u().dense());
            Z_u.emplace_back(sparse_mat());
            Z_u.back().insert(0, 0, nv, nv, sparsity::dense).setRandom();
            dense_zu.push_back(Z_u.back().dense());
        }
        // auto dense = m_data.proj_f_u().dense();
        // REQUIRE(dense.rows() == prob->dim(__y));
        // REQUIRE(dense.cols() == prob->dim(__u)); // Dense matrix dimensions do not match expected values
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        aligned_map_t f_t;
        aligned_map_t f_u_v_diag;
        timed_block_labeled(
            "dense_inner_product",
            auto n = n_trials;
            while (n--) {
                auto &hess = hess_[n];
                // U.noalias() = dense_f_u[n].transpose() * hess * dense_f_u[n];
                dense_inner_product(dense_f_u[n], hess, Q_zz);
            });
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        // timed_block_labeled(
        //     "sparse_inner_product",
        //     auto n = n_trials;
        //     while (n--) {
        //         // auto f = __x;
        //         // auto &jac_sp = m_data[n].proj_f_u();
        //         auto &jac_sp = Z_u[n];
        //         auto &hess = hess_[n];
        //         jac_sp.inner_product(hess, Q_zz);
        //         // Q_zz.noalias() = jac_sp.dense().transpose() * hess * jac_sp.dense();
        //     });
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        timed_block_labeled(
            "sparse_decomposed_inner_product",
            {
                auto n = n_trials;
                while (n--) {
                    // auto f = __x;
                    // auto &jac_sp0 = m_data[n].proj_f_x();//ds_[n].f_y_inv;
                    auto &jac_sp0 = m_data[n].proj_f_u(); // ds_[n].f_y_inv;
                    // auto &jac_sp = m_data[n].approx_[__dyn].jac_[__u];
                    auto &jac_sp = Z_u[n];
                    auto &hess = hess_[n];
                    jac_sp0.inner_product(hess, U);

                    // jac_sp.inner_product(U, Q_zz);
                    dense_inner_product(jac_sp.dense_panels_[0].data_, U, Q_zz);
                    // Q_zz.noalias() = jac_sp0.dense_panels_[0].data_.transpose() * U * jac_sp0.dense_panels_[0].data_;
                }
            });
    }
}