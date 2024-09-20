#include <manbo/core/expression_base.hpp>
#include <manbo/core/shooting_node.hpp>
#include <functional>
#include <iostream>
void nice() {
    auto a = manbo::firstApprox("test", 3);
    std::cout << magic_enum::enum_name(static_cast<manbo::exprBase*>(&a)->type());
    auto b = manbo::problemFormulation();
    // std::bind(&manbo::firstApprox::compute_jacobian, &a, std::placeholders::_1, std::placeholders::_2);
}