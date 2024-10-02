#include <atri/core/expression_base.hpp>
#include <atri/ocp/shooting_node.hpp>
#include <atri/core/multivariate.hpp>
#include <functional>
#include <iostream>
void nice() {
    auto q = std::make_shared<atri::sym>("q", 3, atri::field_type::x);
    auto a = atri::foot_kin_constr("left", q);
    std::cout << magic_enum::enum_name(static_cast<atri::multivariate*>(&a)->approx_level());
    auto b = atri::problem();
    // std::bind(&atri::firstApprox::compute_jacobian, &a, std::placeholders::_1, std::placeholders::_2);
}