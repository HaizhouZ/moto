#include <atri/core/expr.hpp>
#include <atri/ocp/shooting_node.hpp>
#include <atri/solver/ns_sqp.hpp>
#include <functional>
#include <iostream>
void nice() {
    auto q = std::make_shared<atri::sym>("q", 3, atri::__x);
    auto a = atri::foot_kin_constr("left", q);
    std::cout << magic_enum::enum_name(
        static_cast<atri::approx *>(&a)->order());
    auto b = std::make_shared<atri::problem>();
    auto &mem = atri::data_mgr::get<atri::node_data>();
    auto node = atri::shooting_node(atri::problem_ptr_t(), mem);
    // atri::data_mgr::make_data(b);
    // std::bind(&atri::firstApprox::compute_jacobian, &a,
    // std::placeholders::_1, std::placeholders::_2);
}