import moto
import casadi as cs
import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin

from example_robot_data import load


class pinCasadiModel(cpin.Model):
    def __init__(self, model: pin.Model, name: str = "", dt: cs.SX | float = 0.01):
        super().__init__(model)
        if name:
            model.name = name

        # make_primal
        self.q, self.qn = moto.states("q", self.nq)
        self.v, self.vn = moto.states("v", self.nv)
        self.a = moto.inputs("a", self.nv)

        self.is_floating_based: bool = self.nv > self.njoints
        self.ntq = model.nv - 6 if self.is_floating_based else model.nv
        self.tq = moto.inputs("tq", self.ntq)
        self.data = self.createData()
        self.dt = dt


dt = moto.inputs("dt", 1)
go2 = load("go2", display=True, verbose=True)


model = pin.buildModelFromUrdf(go2.urdf, pin.JointModelFreeFlyer())
model = pinCasadiModel(model, dt=dt)

foot_frames = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
foot_idx = [model.getFrameId(f) for f in foot_frames]

cpin.forwardKinematics(model, model.data, model.q, model.v)
cpin.computeJointJacobians(model, model.data)
cpin.updateFramePlacements(model, model.data)

z_f = cs.vcat([model.data.oMf[f].translation[2] for f in foot_idx])
v_f = cs.hcat([cpin.getFrameVelocity(model, model.data, f, pin.LOCAL_WORLD_ALIGNED).linear for f in foot_idx])
foot_jacs = [cpin.getFrameJacobian(model, model.data, f, pin.LOCAL_WORLD_ALIGNED)[:3, :] for f in foot_idx]

# kinematics constraint
active_foot = moto.params("af", len(foot_idx))
k_f = moto.params("k_f")  # kinematics constraint gain

def make_foot_kin_constr(i: int):
    return moto.constr(f"kin_{foot_frames[i]}", [model.q, model.v, k_f, active_foot], cs.vcat([v_f[:2, i], k_f * z_f[i] + v_f[2, i]]) * active_foot[i])
kin_constr = [make_foot_kin_constr(i) for i in range(len(foot_frames))]

# floating base inverse dynamics
f_f = [moto.inputs(f"f_{f}", 3) for f in foot_frames]
inv_dyn = cpin.rnea(model, model.data, model.q, model.v, model.a)
F_f = [foot_jacs[i].T @ f_f[i] for i in range(len(foot_frames))]
floating_base_inv_dyn = (inv_dyn - sum(F_f))[:6]

# friction cone constraints
mu = moto.params("mu")  # friction coefficient


def make_fric_cone(i, f: moto.sym):
    cone = cs.vcat(
        [
            f[0] - mu * f[2],
            -f[0] - mu * f[2],
            f[1] - mu * f[2],
            -f[1] - mu * f[2],
        ]
    )
    return moto.constr(f"fric_{foot_frames[i]}", [f, mu], cone).as_ineq()


fric = [make_fric_cone(i, f) for i, f in enumerate(f_f)]

# foot force contact constraints
def make_contact_constr(i: int, f: moto.sym):
    return moto.constr(f"contact_{foot_frames[i]}", [active_foot, f], (1 - active_foot[i]) * f)

contact = [make_contact_constr(i, f) for i, f in enumerate(f_f)]

# state_cost = 

prob = moto.ocp.create()
prob.add(kin_constr + fric + contact)