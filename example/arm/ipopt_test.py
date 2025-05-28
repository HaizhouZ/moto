import pinocchio as pin
import pinocchio.casadi as cpin
import casadi as cs
import numpy as np

model_path = "rsc/iiwa_description/urdf"
urdf_filename = "iiwa14.urdf"
urdf_path = f"{model_path}/{urdf_filename}"


# Build the Pinocchio model
model = pin.buildModelFromUrdf(urdf_path)
data = model.createData()

# Convert to CasADi Pinocchio model
cpin_model = cpin.Model(model)
cpin_data = cpin_model.createData()

def make_dyn():
    q = cs.SX.sym("q", model.nq)
    v = cs.SX.sym("v", model.nv)
    qn = cs.SX.sym("qn", model.nq)
    vn = cs.SX.sym("vn", model.nv)
    tq = cs.SX.sym("tq", model.nv)

    dt = 0.01
    impl_euler = qn - (q + vn * dt)
    invdyn = cpin.rnea(cpin_model, cpin_data, q, v, (vn - v) / dt) - tq

    return cs.Function('dyn', [q, v, qn, vn, tq], [impl_euler, invdyn])

dyn = make_dyn()

def make_cost():
    q = cs.SX.sym("q", model.nq)
    pos_d = cs.SX.sym("pos_d", 3)
    cpin.forwardKinematics(cpin_model, cpin_data, q)
    cpin.updateFramePlacements(cpin_model, cpin_data)
    r_ee = cpin_data.oMf[model.getFrameId("iiwa_link_ee_kuka")].translation
    kin_cost = cs.sumsqr(r_ee - pos_d)

    return cs.Function('cost', [q, pos_d], [kin_cost])

kin_cost = make_cost()

sqp = cs.Opti()
sqp.solver("ipopt", {}, {})


q0 = sqp.parameter(model.nq)
v0 = sqp.parameter(model.nv)
pos_d = sqp.parameter(3)

q_ = [q0]
v_ = [v0]

sqp.set_value(pos_d, 0.5)
sqp.set_value(q0, np.ones(model.nq))
sqp.set_value(v0, np.zeros(model.nv))
cost = 0.
cost += 0.5 * (10 * cs.sumsqr(q0) + 0.1 * cs.sumsqr(v0))
N = 5
for i in range(N):
    q = sqp.variable(model.nq)
    v = sqp.variable(model.nv)
    tq = sqp.variable(model.nv)
    euler, rnea = dyn(q_[-1], v_[-1], q, v, tq)
    cost_ = kin_cost(q, pos_d)
    sqp.subject_to(euler == 0)
    sqp.subject_to(rnea == 0)
    cost += 100 * cost_
    cost += 0.5 * (10 * cs.sumsqr(q) + 0.1 * cs.sumsqr(v) + 1e-2 * cs.sumsqr(tq))
    q_.append(q)
    v_.append(v)

sqp.minimize(cost)
sol = sqp.solve()
print(sol.value(cost))

pin.forwardKinematics(model, data, sol.value(q_[-1]))
pin.updateFramePlacements(model, data)
r_ee = data.oMf[model.getFrameId("iiwa_link_ee_kuka")].translation
print(sol.value(q_[-1]))
print(r_ee)

