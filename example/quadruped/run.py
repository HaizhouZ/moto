import moto
import casadi as cs
import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin

from example_robot_data import load


class pinCasadiModel(cpin.Model):
    def __init__(self, model: pin.Model, name: str = "", dt: cs.SX | float = 0.01, q_nom: np.ndarray | None = None):
        super().__init__(model)
        if name:
            model.name = name

        # make_primal
        self.q, self.qn = moto.states("q", self.nq, q_nom)
        self.v, self.vn = moto.states("v", self.nv)
        self.a = moto.inputs("a", self.nv)

        self.is_floating_based: bool = self.nv > self.njoints
        self.ntq = model.nv - 6 if self.is_floating_based else model.nv
        self.tq = moto.inputs("tq", self.ntq)
        self.data = self.createData()
        self.dt = dt


dt = moto.inputs("dt", 1, default_val=0.01)
go2 = load("go2", display=True, verbose=True)


model = pin.buildModelFromUrdf(go2.urdf, pin.JointModelFreeFlyer())
model = pinCasadiModel(model, dt=dt, q_nom=go2.q0)

foot_frames = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
foot_idx = [model.getFrameId(f) for f in foot_frames]

cpin.forwardKinematics(model, model.data, model.q, model.v)
cpin.computeJointJacobians(model, model.data)
cpin.updateFramePlacements(model, model.data)

z_f = cs.vcat([model.data.oMf[f].translation[2] for f in foot_idx])
v_f = cs.hcat([cpin.getFrameVelocity(model, model.data, f, pin.LOCAL_WORLD_ALIGNED).linear for f in foot_idx])
foot_jacs = [cpin.getFrameJacobian(model, model.data, f, pin.LOCAL_WORLD_ALIGNED)[:3, :] for f in foot_idx]

# kinematics constraint
active_foot = moto.params("af", len(foot_idx), default_val=1)
k_f = moto.params("k_f", default_val=100)  # kinematics constraint gain


def make_foot_kin_constr(i: int):
    return moto.constr(
        f"kin_{foot_frames[i]}",
        [model.q, model.v, k_f, active_foot],
        cs.vcat([v_f[:2, i], k_f * z_f[i] + v_f[2, i]]) * active_foot[i],
    )


kin_constr = [make_foot_kin_constr(i) for i in range(len(foot_frames))]

# floating base inverse dynamics
f_f = [moto.inputs(f"f_{f}", 3) for f in foot_frames]
inv_dyn = cpin.rnea(model, model.data, model.q, model.v, model.a)
F_f = [foot_jacs[i].T @ f_f[i] for i in range(len(foot_frames))]
floating_base_inv_dyn = (inv_dyn * dt - sum(F_f))[:6]
fb_id_constr = moto.constr("fb_id", [model.q, model.v, model.a, dt, *f_f], floating_base_inv_dyn)

# friction cone constraints
mu = moto.params("mu", default_val=0.7)  # friction coefficient


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

# timestep constraint
dt_constr = moto.constr("dt", [dt], cs.vcat([1e-4 - dt, dt - 3e-2])).as_ineq()


# implicit euler
def implicit_euler():
    q_next = cpin.integrate(model, model.q, model.vn * dt)
    v_next = model.v + model.a * dt
    return moto.constr(
        "euler", [model.q, model.v, model.qn, model.vn, model.a, dt], cs.vcat([model.qn - q_next, model.vn - v_next])
    )


q_nom = moto.params("q_nom", model.nq, default_val=go2.q0)

state_cost = 1 * cs.sumsqr(model.q - q_nom) + 1e-2 * cs.sumsqr(model.v)
input_cost = 1e-4 * cs.sumsqr(model.a) + 1e-2 * cs.sumsqr(cs.vcat(f_f))

running_cost = moto.cost("c", [model.q, model.v, model.a, q_nom, *f_f, dt], (state_cost + input_cost) * dt)
timing_cost = moto.cost("c_t", [dt], 10 * cs.sumsqr(dt - 1e-2))
terminal_cost = moto.cost("c", [model.q, model.v, q_nom], state_cost).as_terminal()

prob = moto.ocp.create()
prob.add(kin_constr + fric + contact)
prob.add(implicit_euler())
prob.add(fb_id_constr)
prob.add([running_cost, timing_cost])
prob.add(dt_constr)

prob_term = prob.clone()
prob_term.add(terminal_cost)

moto.print_problem(prob)
print("--" * 20)
# moto.print_problem(prob_term)
# exit(0)

sqp = moto.sqp(n_job=1)
g = sqp.graph
n0 = g.set_head(g.add(sqp.create_node(prob)))
n1 = g.set_tail(g.add(sqp.create_node(prob_term)))
g.add_edge(n0, n1, 100)

sqp.update(10)
