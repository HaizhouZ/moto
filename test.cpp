#include <atri/core/expr.hpp>
#include <atri/ocp/shooting_node.hpp>
#include <functional>
#include <iostream>
void nice() {
    auto q = std::make_shared<atri::sym>("q", 3, atri::field_type::x);
    auto a = atri::foot_kin_constr("left", q);
    std::cout << magic_enum::enum_name(static_cast<atri::approximation*>(&a)->approx_level());
    auto b = std::make_shared<atri::expr_collection>();
    auto c = atri::mem_mgr::get();
    atri::mem_mgr::make_approx_data(b);
    // std::bind(&atri::firstApprox::compute_jacobian, &a, std::placeholders::_1, std::placeholders::_2);
}