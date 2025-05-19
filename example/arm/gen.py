import pinocchio as pin
import pinocchio.casadi as cpin
import casadi as cs

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

q = cs.SX.sym("q", model.nq)
v = cs.SX.sym("v", model.nv)
vn = cs.SX.sym("vn", model.nv)
tq = cs.SX.sym("tq", model.nv)

tau = cpin.rnea(cpin_model, cpin_data, q, v, (vn - v) / 0.01) - tq

from atri.codegen import *

generate_and_compile(
    "rnea",
    [q, v, vn, tq],
    [tau],
    "gen",
)

generate_and_compile(
    "rnea_jac",
    [q, v, vn, tq],
    [cs.jacobian(tau, q), cs.jacobian(tau, v), cs.jacobian(tau, vn), cs.jacobian(tau, tq)],
    "gen",
)

pos_d = cs.SX.sym("pos_d", 3)

cpin.forwardKinematics(cpin_model, cpin_data, q)
cpin.updateFramePlacements(cpin_model, cpin_data)

kin_cost = cs.sumsqr(cpin_data.oMf[model.getFrameId('iiwa_link_ee_kuka')].translation - pos_d)

generate_and_compile(
    "kin_cost",
    [q, pos_d],
    [kin_cost],
    "gen",
)

generate_and_compile(
    "kin_cost_jac",
    [q, pos_d],
    [cs.jacobian(kin_cost, q)],
    "gen",
)

generate_and_compile(
    "kin_cost_hess",
    [q, pos_d],
    [cs.hessian(kin_cost, q)[0]],
    "gen",
)

wait_until_generated()