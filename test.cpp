#include <manbo/core/expression_base.hpp>
#include <manbo/ocp/shooting_node.hpp>
#include <manbo/core/multivariate.hpp>
#include <functional>
#include <iostream>
void nice() {
    auto q = std::make_shared<manbo::symbolic>("q", 3, manbo::field_type::x);
    auto a = manbo::foot_kin_constr("left", q);
    std::cout << magic_enum::enum_name(static_cast<manbo::multivariate*>(&a)->approx_level());
    auto b = manbo::problem_t();
    // std::bind(&manbo::firstApprox::compute_jacobian, &a, std::placeholders::_1, std::placeholders::_2);
}