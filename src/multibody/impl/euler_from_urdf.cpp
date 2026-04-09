
// #undef PINOCCHIO_ENABLE_TEMPLATE_INSTANTIATION

// #include <moto/multibody/impl/euler.hpp>
// #include <pinocchio/multibody/model.hpp>
// #include <pinocchio/parsers/urdf.hpp>
// #include <pinocchio/autodiff/casadi.hpp>

// namespace pin = pinocchio;
// namespace moto {
// namespace multibody {
// func euler::from_urdf(const std::string &urdf_path, var dt, root_joint_t root_joint, euler::v_int_type v_int) {
// #ifdef MOTO_WITH_PINOCCHIO
//     pin model;
//     if (root_joint == root_joint_t::fixed) {
//         pin::urdf::buildModel(urdf_path, model);
//     } else {
//         pin::JointModelComposite root(2);
//         root.addJoint(pin::JointModelTranslation());
//         switch (root_joint) {
//         case root_joint_t::xyz_quat:
//             root.addJoint(pin::JointModelSpherical());
//             break;
//         case root_joint_t::xyz_eulerZYX:
//             root.addJoint(pin::JointModelSphericalZYX());
//             break;
//         default:
//             throw std::runtime_error("Unknown root joint type");
//         }
//         pin::urdf::buildModel(urdf_path, root, model);
//     }

//     using cpin_model = pin::ModelTpl<casadi::SX>;
//     cpin_model cs_model(model);
//     auto [q, qn] = sym::states(model.name + "_q", model.nq);
//     auto [v, vn] = sym::states(model.name + "_v", model.nv);
//     auto a = sym::inputs(model.name + "_a", model.nv);
//     auto get_pos_step = [&]() {
//         switch (v_int) {
//         case v_int_type::_explicit:
//             return v * dt;
//         case v_int_type::_mid_point:
//             return (vn + v) / 2 * dt;
//         case v_int_type::_implicit:
//             return vn * dt;
//         default:
//             throw std::runtime_error("Unknown velocity integration type");
//         }
//     };
//     auto pos_step = get_pos_step();
//     using pin_cs_t = Eigen::Matrix<casadi::SX, Eigen::Dynamic, 1>;
//     using pin_cs_m = Eigen::Matrix<casadi::SX, Eigen::Dynamic, Eigen::Dynamic>;
//     pin_cs_t pos_step_s(model.nv);
//     pin::casadi::copy(pos_step, pos_step_s);
//     pin_cs_t qs(model.nq), qns(model.nq), vs(model.nv), vns(model.nv), as(model.nv);
//     pin::casadi::copy(q, qs);
//     pin::casadi::copy(qn, qns);
//     pin::casadi::copy(v, vs);
//     pin::casadi::copy(vn, vns);
//     pin::casadi::copy(a, as);
//     // auto pos_int_only_s = pin::integrate(cs_model, qs, vs * dt);
//     auto pos_int_s = pin::integrate(cs_model, qs, pos_step_s);
//     cs::SX pos_int, pos_int_only;
//     pin::casadi::copy(pos_int_s, pos_int);
//     pin::casadi::copy(pos_int_only_s, pos_int_only);
//     auto pos_diff_s = pin::difference(cs_model, pos_int_s, qns);
//     cs::SX pos_diff;
//     pin::casadi::copy(pos_diff_s, pos_diff);
//     pin_cs_m dpos_int_dq(model.nv, model.nv);
//     dpos_int_dq.setZero();
//     pin_cs_m dpos_int_dqstep(model.nv, model.nv);
//     dpos_int_dqstep.setZero();
//     pin::dIntegrate(cs_model, qs, pos_step_s, dpos_int_dq, pin::ARG0);
//     pin::dIntegrate(cs_model, qs, pos_step_s, dpos_int_dqstep, pin::ARG1);
//     cs::SX dpos_int_dq_cs = cs::SX::zeros(model.nv, model.nv);
//     cs::SX dpos_int_dqstep_cs = cs::SX::zeros(model.nv, model.nv);
//     pin::casadi::copy(dpos_int_dq, dpos_int_dq_cs);
//     pin::casadi::copy(dpos_int_dqstep, dpos_int_dqstep_cs);
//     pin_cs_m dpos_diff_dq(model.nv, model.nv);
//     dpos_diff_dq.setZero();
//     pin_cs_m dpos_diff_dqn(model.nv, model.nv);
//     dpos_diff_dqn.setZero();
//     pin::dDifference(cs_model, pos_int_s, qs, dpos_diff_dqn, pin::ARG1);
//     pin::dDifference(cs_model, pos_int_s, qs, dpos_diff_dq, pin::ARG0);
//     cs::SX dpos_diff_dq_cs, dpos_diff_dqn_cs;
//     pin::casadi::copy(dpos_diff_dq, dpos_diff_dq_cs);
//     pin::casadi::copy(dpos_diff_dqn, dpos_diff_dqn_cs);
//     std::string name = model.name + std::string(magic_enum::enum_name<euler::v_int_type>(v_int)) + "_euler";
//     return func(multibody::euler(name, state{q, v, a, dt, root_joint},
//                                  pos_step, pos_diff,
//                                  std::array{dpos_diff_dqn_cs, dpos_diff_dq_cs},
//                                  pos_int_only, std::array{dpos_int_dq_cs, dpos_int_dqstep_cs}));
// #else
//     throw std::runtime_error("Pinocchio is required to load euler from URDF.");
//     return func();
// #endif
// }
// } // namespace multibody
// } // namespace moto