// #include <moto/multibody/lifted_contact_invdyn.hpp>
// #include <moto/multibody/pinocchio_utils.hpp>
// #include <pinocchio/multibody/model.hpp>
// #include <pinocchio/parsers/urdf.hpp>

// namespace moto {
// namespace multibody {

// void lifted_contact_invdyn::load_model(const std::string &urdf_path, root_joint_t root, bool verbose) {
//     if (!model_) {
//         model_ = new pin_model();
//     }
//     auto parse_model = [&](auto &&root_joint) {
//         pin::urdf::buildModelFromXML(urdf_path, root_joint, *model_, verbose);
//     };
//     switch (root) {
//     case root_joint_t::xyz_quat: {
//         parse_model(pin::JointModelFreeFlyer());
//         break;
//     }
//     case root_joint_t::xyz_eulerZYX: {
//         pin::JointModelComposite composite_root(2);
//         composite_root.addJoint(pin::JointModelTranslation());
//         composite_root.addJoint(pin::JointModelSphericalZYX());
//         parse_model(composite_root);
//         break;
//     }
//     case root_joint_t::fixed: {
//         pin::urdf::buildModelFromXML(urdf_path, *model_, verbose);
//         break;
//     }
//     }
// }
// } // namespace multibody
// } // namespace moto