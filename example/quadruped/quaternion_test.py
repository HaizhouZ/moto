import moto
import casadi as cs
import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin
from scipy.spatial.transform.rotation import Rotation as R


urdf = (
    ""
    '<?xml version="1.0"?>\n'
    '<robot name="box">\n'
    '  <link name="base_link">\n'
    "    <inertial>\n"
    '      <origin xyz="0 0 0" rpy="0 0 0"/>\n'
    '      <mass value="1.0"/>\n'
    '      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>\n'
    "    </inertial>\n"
    "  </link>\n"
    "</robot>\n"
)

model = pin.buildModelFromXML(urdf, pin.JointModelSpherical())
r_model = model
model = cpin.Model(r_model)  # casadi model

q, qn = moto.quaternion.create("box_quat")

q.finalize()
qn.finalize()
print("q, qn finalized")


n_trials = 1000
for _ in range(n_trials):
    dt = float(np.random.random() * 0.1)
    q0 = R.random().as_quat()
    # print("q0", q0)
    w = np.random.random(3)
    q1 = qn.integrate(q0, w, dt)
    gt = pin.integrate(r_model, q0, w * dt)
    if not np.allclose(q1, gt, atol=1e-8):
        raise RuntimeError("mismatch")
    w_diff = qn.difference(q1, q0) / dt
    if not np.allclose(w_diff, w, atol=1e-8):
        raise RuntimeError("mismatch difference")
    # print("q1", q1)
    # print("gt", pin.integrate(r_model, q0, w * 0.01))

print("all tests passed")
