#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <moto/ocp/problem.hpp>
#include <moto/utils/codegen.hpp>
#define ENABLE_TIMED_BLOCK
#include <moto/multibody/stacked_euler.hpp>
#include <moto/utils/timed_block.hpp>

#include <moto/core/parallel_job.hpp>

#include <moto/multibody/lcid.hpp>
#include <moto/multibody/lcid_riccati/lcid_solver.hpp>
#include <moto/ocp/cost.hpp>
#include <moto/solver/ns_sqp.hpp>
#include <moto/spmm/sparse_mat.hpp>

int main() {
    using namespace moto;
    using dyn_t = multibody::stacked_euler;
    auto dt = sym::inputs("dt", 1, 0.01);
    constr dyn = dyn_t("stacked_euler");
    auto e = multibody::euler::from_urdf("rsc/go2_description/urdf/go2_description.urdf", dt);//, multibody::root_joint_t::xyz_eulerZYX, multibody::euler::v_int_type::_explicit);
    auto b = multibody::euler::from_urdf("rsc/go2_description/urdf/box_description.urdf", dt); //, multibody::root_joint_t::xyz_eulerZYX);//, multibody::euler::v_int_type::_explicit);
    dyn.as<dyn_t>().add(e);
    // dyn.as<dyn_t>().add(b);
    // auto bb = b.as<multibody::euler>().share();
    // dyn.as<dyn_t>().add(bb);

    auto &e_ = e.as<multibody::euler>();
    // auto &b_ = b.as<multibody::euler>();
    // auto &bb_ = bb.as<multibody::euler>();

    auto f_e = sym::inputs("f_e", e_.v->dim());
    // auto f_b = sym::inputs("f_b", b_.v->dim());
    // auto f_c = sym::inputs("f_c", bb_.v->dim());
    auto tq_e = sym::inputs("tq_e", e_.v->dim() - 6);

    auto test_kin_constr = constr("test_kin_constr", {e_.q, e_.v, e_.a}, e_.v + e_.a, approx_order::first);
    auto test_dyn_constr = constr("test_dyn_constr", {e_.q, e_.v, e_.a, tq_e, f_e}, e_.a + e_.v - f_e + cs::SX::vertcat({cs::SX::zeros(6), tq_e}), approx_order::first);
    // auto test_kin_constr2 = constr("test_kin_constr2", {b_.q, b_.v, b_.a}, b_.v + b_.a, approx_order::first);
    // auto test_dyn_constr2 = constr("test_dyn_constr2", {b_.q, b_.v, b_.a, f_b}, b_.a + b_.v - f_b, approx_order::first);
    // auto test_kin_constr3 = constr("test_kin_constr3", {bb_.q, bb_.v, bb_.a}, bb_.v + bb_.a, approx_order::first);
    // auto test_dyn_constr3 = constr("test_dyn_constr3", {bb_.q, bb_.v, bb_.a, f_c}, bb_.a + bb_.v - f_c, approx_order::first);

    custom_func lcid(multibody::lcid("lcid_test"));
    lcid.as<multibody::lcid>().add_kin_constr({e_}, {}, test_kin_constr);
    lcid.as<multibody::lcid>().add_dyn_constr({e_}, {f_e}, test_dyn_constr);
    // lcid.as<multibody::lcid>().add_dyn_constr({b_}, {f_b}, test_dyn_constr2);
    // lcid.as<multibody::lcid>().add_kin_constr({b_}, {}, test_kin_constr2);
    // lcid.as<multibody::lcid>().add_dyn_constr({bb_}, {f_c}, test_dyn_constr3);
    // lcid.as<multibody::lcid>().add_kin_constr({bb_}, {}, test_kin_constr3);

    cost c("test_c", {e_.q, e_.v}, cs::SX::sumsqr(cs::SX::vertcat({e_.q, e_.v})));
    c.set_diag_hess();
    cost c2("test_c2", {e_.a, f_e, dt}, cs::SX::sumsqr(cs::SX::vertcat({e_.a, f_e})) + 100 * dt * dt);
    c2.set_diag_hess();
    cost c_t = c->clone().as_terminal();

    auto prob = ocp::create();
    prob->add(dt);
    prob->add(dyn);
    prob->add(lcid);
    prob->add(c);
    prob->add(c2);
    prob->print_summary();
    auto prob_term = prob->clone();
    prob_term->add(c_t);
    using solver_t = solver::lcid_riccati::lcid_solver;
    solver_t solver(lcid, dyn);

    struct MOTO_ALIGN_NO_SHARING node_t : public node_data, public solver_t::ns_riccati_data {
        node_t(const ocp_ptr_t &p, solver_t &solver)
            : node_data(p), solver_t::ns_riccati_data(solver.create_data(this)) {}
    };

    size_t N = 100;
    std::vector<node_t> data;
    data.reserve(N);
    for (size_t i = 0; i < N; ++i) {
        if (i == N - 1)
            data.emplace_back(prob_term, solver);
        else
            data.emplace_back(prob, solver);
        for (auto f : primal_fields) {
            data[i].value(f).setRandom();
        }
        for (auto f : constr_fields)
            data[i].dense().dual_[f].setZero();

        data[i].update_approximation();
    }

    size_t n_trials = 1000;
    while (n_trials--) {
        timed_block_start("factorization");
        parallel_for(0, N, [&](size_t i) { solver.ns_factorization(&data[i]); });
        timed_block_end("factorization");
        std::vector<std::pair<node_t *, node_t *>> args;
        args.reserve(N);
        for (int i = N - 1; i >= 0; --i) {
            args.emplace_back(&data[i], i > 0 ? &data[i - 1] : nullptr);
            // // solver.ns_factorization(&data[i]);
            // // if (i > 0)
            // if (i == 0)
            //     solver.riccati_recursion(&data[i], nullptr);
            // else
            //     solver.riccati_recursion(&data[i], &data[i - 1]);
        }
        timed_block_start("riccati");
        sequential_for(0, N, [&](size_t i, size_t j) { solver.riccati_recursion(args[i].first, args[i].second); });
        timed_block_end("riccati");
    }

    return 0;
}