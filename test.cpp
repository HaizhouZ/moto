#include <atri/core/expr.hpp>
#include <atri/ocp/shooting_node.hpp>
#include <atri/solver/csqp.hpp>
#include <functional>
#include <iostream>
void nice() {
    auto q = std::make_shared<atri::sym>("q", 3, atri::field::type::x);
    auto a = atri::foot_kin_constr("left", q);
    std::cout << magic_enum::enum_name(
        static_cast<atri::approx *>(&a)->order());
    auto b = std::make_shared<atri::expr_sets>();
    auto &mem = atri::data_mgr::get<atri::node_data>();
    auto node = atri::shooting_node(atri::expr_sets_ptr_t(), mem);
    // atri::data_mgr::make_data(b);
    // std::bind(&atri::firstApprox::compute_jacobian, &a,
    // std::placeholders::_1, std::placeholders::_2);
}