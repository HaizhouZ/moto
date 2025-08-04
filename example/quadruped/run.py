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
active_foot = [moto.params(f"af_{f}", 1, default_val=1) for f in foot_frames]  # active foot indicator
k_f = moto.params("k_f", default_val=100)  # kinematics constraint gain

z_clip = moto.params("z_c", 1, default_val=0.01)  # minimum height for the foot to be considered in contact


def make_foot_kin_constr(i: int):
    return moto.constr(
        f"kin_{foot_frames[i]}",
        [model.q, model.v, k_f, *active_foot, z_clip],
        cs.vcat([v_f[:2, i], k_f * cs.tanh(z_f[i]) * z_clip + v_f[2, i]]) * active_foot[i],
        # cs.vcat([v_f[:2, i], k_f * z_f[i]  + v_f[2, i]]) * active_foot[i],
    )


kin_constr = [make_foot_kin_constr(i) for i in range(len(foot_frames))]

# floating base inverse dynamics
f_f = [moto.inputs(f"f_{f}", 3, default_val=np.array([0.0, 0.0, 0.1])) for f in foot_frames]
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
    return moto.constr(f"contact_{foot_frames[i]}", [*active_foot, f], (1 - active_foot[i]) * f)


contact = [make_contact_constr(i, f) for i, f in enumerate(f_f)]

# timestep constraint
dt_constr = moto.constr("dt", [dt], cs.vcat([1e-4 - dt, dt - 3e-2])).as_ineq()
# dt_constr = moto.constr("dt_fix", [dt], dt - 1e-2)


# implicit euler
def implicit_euler():
    q_next = cpin.integrate(model, model.q, model.vn * dt)
    v_next = model.v + model.a * dt
    return moto.constr(
        "euler", [model.q, model.v, model.qn, model.vn, model.a, dt], cs.vcat([model.qn - q_next, model.vn - v_next])
    )


q_d = np.copy(go2.q0)

q_nom = moto.params("q_nom", model.nq, default_val=q_d)

q_nom_res = model.q - q_nom

state_cost = 10.0 * cs.sumsqr(q_nom_res[:7]) + 1.0 * cs.sumsqr(q_nom[7:]) + 1e-2 * cs.sumsqr(model.v)
input_cost = 1e-4 * cs.sumsqr(model.a) + 1e-2 * cs.sumsqr(cs.vcat(f_f))

running_cost = moto.cost("c", [model.q, model.v, model.a, q_nom, *f_f], (state_cost + input_cost))
timing_cost = moto.cost("c_t", [dt], 1000 * cs.sumsqr(dt - 1e-2))
terminal_cost = moto.cost("c", [model.q, model.v, q_nom], state_cost).as_terminal()

prob = moto.ocp.create()
prob.add(kin_constr + fric + contact)
prob.add(implicit_euler())
prob.add(dt_constr)
prob.add(fb_id_constr)
prob.add([running_cost, timing_cost])

prob_term = prob.clone()
prob_term.add(terminal_cost)

moto.print_problem(prob)
print("--" * 20)
# moto.print_problem(prob_term)
# exit(0)

N_horizon = 100

sqp = moto.sqp(n_job=1)
g = sqp.graph
n0 = g.set_head(g.add(sqp.create_node(prob)))
n1 = g.set_tail(g.add(sqp.create_node(prob_term)))
g.add_edge(n0, n1, N_horizon)

sqp.settings.mu_method = moto.sqp.adaptive_mu_t.mehrotra_probing
# sqp.settings.mu_method = moto.sqp.adaptive_mu_t.mehrotra_predictor_corrector
# sqp.settings.ipm_conditional_corrector = True


# setup gait
steps = 2
nodes_per_step = 40
total_gait_steps = steps * nodes_per_step
stance_length = int(N_horizon - total_gait_steps) / 2
step = 0
node_idx = 0


def gait_setup(data: moto.sqp.data_type):
    global step, node_idx
    if node_idx >= stance_length and node_idx + stance_length < N_horizon:
        switch_step  = (node_idx - stance_length) % nodes_per_step == 0
        if switch_step:
            step += 1
        if step % 2 == 0:
            data.value[active_foot[0]] = 1
            data.value[active_foot[3]] = 1
            data.value[active_foot[1]] = 0
            data.value[active_foot[2]] = 0
        elif step % 2 == 1:
            data.value[active_foot[0]] = 0
            data.value[active_foot[3]] = 0
            data.value[active_foot[1]] = 1
            data.value[active_foot[2]] = 1
    else:
        data.value[active_foot[0]] = 1
        data.value[active_foot[1]] = 1
        data.value[active_foot[2]] = 1
        data.value[active_foot[3]] = 1
    print(node_idx, data.value[active_foot[0]], data.value[active_foot[1]], data.value[active_foot[2]], data.value[active_foot[3]], data.prob.uid)
    node_idx += 1


sqp.apply_forward(gait_setup)
exit(0)
sqp.update(6)

q_res = []
dt_res = []
node_idx = 0


def get_sym(node: moto.sqp.data_type):
    global node_idx
    q_res.append(node.value[model.q])
    dt_res.append(node.value[dt])
    node_idx += 1
    if node_idx >= N_horizon:
        q_res.append(node.value[model.qn])


sqp.apply_forward(get_sym)

import time

while True:
    for i in range(len(q_res)):
        start = time.perf_counter()
        go2.display(q_res[i])
        if i is not N_horizon:
            dt_ = dt_res[i]
            while time.perf_counter() - start < dt_:
                pass
    time.sleep(0.5)

# def print_sym(node: moto.sqp.data_type):
#     node.sym.print()
#     node.print_residuals()

# sqp.apply_forward(print_sym, early_stop=20)
