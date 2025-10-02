import moto
import casadi as cs
import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin

# from example_robot_data import load

urdf = "/home/harper/Documents/moto/rsc/go2_description/urdf/box_description.urdf"
# root_joint = pin.JointModelComposite()
# root_joint.addJoint(pin.JointModelTranslation())
# root_joint.addJoint(pin.JointModelSpherical())
# root_joint = pin.JointModelFreeFlyer()
model = pin.buildModelFromUrdf(urdf, pin.JointModelSpherical())# root_joint)
r_model = model
model = cpin.Model(r_model)  # casadi model

q, qn = moto.quaternion.create('box_quat')

qn.sym_handle.finalize()

from scipy.spatial.transform.rotation import Rotation as R

n_trials = 1000
for _ in range(n_trials):
    dt = float(np.random.random() * 0.1)
    q0 = R.random().as_quat()
    # print("q0", q0)
    w = np.random.random(3)
    q1 = qn.sym_handle.integrate(q0, w, dt)
    gt = pin.integrate(r_model, q0, w * dt)
    if not np.allclose(q1, gt, atol=1e-8):
        raise RuntimeError("mismatch")
    w_diff = qn.sym_handle.difference(q1, q0) / dt
    if not np.allclose(w_diff, w, atol=1e-8):
        raise RuntimeError("mismatch difference")
    # print("q1", q1)
    # print("gt", pin.integrate(r_model, q0, w * 0.01))

print("all tests passed")