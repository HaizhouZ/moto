#include <atri/core/expr.hpp>
#include <atri/ocp/shooting_node.hpp>
#include <functional>
#include <iostream>
#include <atri/solver/cslq.hpp>
void nice() {
    auto q = std::make_shared<atri::sym>("q", 3, atri::field::type::x);
    auto a = atri::foot_kin_constr("left", q);
    std::cout << magic_enum::enum_name(static_cast<atri::approx*>(&a)->order());
    auto b = std::make_shared<atri::expr_sets>();
    auto c = atri::data_mgr::get();
    atri::data_mgr::make_data(b);
    // std::bind(&atri::firstApprox::compute_jacobian, &a, std::placeholders::_1, std::placeholders::_2);
}