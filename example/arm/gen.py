import pinocchio as pin
import pinocchio.casadi as cpin
import casadi as cs
import atri

import atri.codegen

# Load the robot model
model_path = "rsc/iiwa_description/urdf"
urdf_filename = "iiwa14.urdf"
urdf_path = f"{model_path}/{urdf_filename}"

# Build the Pinocchio model
model = pin.buildModelFromUrdf(urdf_path)
data = model.createData()

# Convert to CasADi Pinocchio model
cpin_model = cpin.Model(model)
cpin_data = cpin_model.createData()


q = atri.sym("q", model.nq, atri.field_x)
v = atri.sym("v", model.nv, atri.field_x)
qn = atri.sym("qn", model.nq, atri.field_y)
vn = atri.sym("vn", model.nv, atri.field_y)
tq = atri.sym("tq", model.nv, atri.field_u)

dt = 0.01
impl_euler = qn - (q + vn * dt)
invdyn = cpin.rnea(cpin_model, cpin_data, q, v, (vn - v) / dt) - tq

# a = cs.SX.sym("a", model.nv)
a = (vn - v) / dt

rnea_dq, rnea_dv, rnea_da = cpin.computeRNEADerivatives(cpin_model, cpin_data, q, v, a)

# rnea_dq = cs.substitute(rnea_dq, a, (vn - v) / dt)
rnea_dv = rnea_dv - rnea_da / dt
rnea_dvn = rnea_da / dt

in_args = [q, v, qn, vn, tq]

jac_pos = [(e, cs.jacobian(impl_euler, e)) for e in in_args]
jac_vel = []
jac_vel.append((q, rnea_dq))
jac_vel.append((v, rnea_dv))
jac_vel.append((vn, rnea_dvn))
jac_vel.append((tq, -cs.SX_eye(model.nv)))


atri.generate_and_compile(
    "euler",
    [q, qn, vn],
    impl_euler,
    "gen",
    ext_jac=jac_pos,
)

atri.generate_and_compile(
    "rnea",
    [q, v, vn, tq],
    invdyn,
    "gen",
    ext_jac=jac_vel,
)

pos_d = atri.sym("pos_d", 3, atri.field_p)

cpin.forwardKinematics(cpin_model, cpin_data, q)
cpin.updateFramePlacements(cpin_model, cpin_data)

r = cs.SX.sym("r", 3)
r_ee = cpin_data.oMf[model.getFrameId("iiwa_link_ee_kuka")].translation
cpin.computeJointJacobians(cpin_model, cpin_data)
J = cpin.getFrameJacobian(
    cpin_model, cpin_data, model.getFrameId("iiwa_link_ee_kuka"), pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
)
kin_cost = cs.sumsqr(r_ee - pos_d)
kin_cost_proxy = cs.sumsqr(r - pos_d)


kin_cost_jac = cs.substitute(cs.jacobian(kin_cost_proxy, r), r, r_ee) @ J[:3, :]
kin_cost_hess = cs.jacobian(kin_cost_jac, q)


atri.generate_and_compile(
    "kin_cost",
    [q, pos_d],
    kin_cost,
    "gen",
    exclude=[pos_d],
    ext_jac=[(q, kin_cost_jac)],
    ext_hess=[(q, q, kin_cost_hess)],
)

atri.wait_until_generated()
