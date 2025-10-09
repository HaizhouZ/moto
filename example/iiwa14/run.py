import moto
import casadi as cs
import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin

from example_robot_data import load


class pinCasadiModel(cpin.Model):
    def __init__(
        self,
        model: pin.Model,
        name: str = "",
        dt: cs.SX | float = 0.01,
        q_nom: np.ndarray | None = None,
        dense: bool = True,
        foot_frames: list[str] | None = None,
        use_fwd_dyn: bool = False,
    ):
        self.fmodel = model
        super().__init__(model)
        if name:
            model.name = name
        else:
            name = model.name.replace("_description", "")
            self.name = name
        self.is_floating_based: bool = self.nv > self.njoints
        self.nj = self.nv - 6 if self.is_floating_based else self.nv
        self.data = self.createData()
        self.dt = dt
        self.dense = dense
        self.use_fwd_dyn = use_fwd_dyn
        self.ee_frame = "ee_link"
        self.ee_id = model.getFrameId(self.ee_frame)

        if self.is_floating_based:
            self.nqb = 6
        if self.nq - self.nv == 1:
            self.nqb = 7

        if dense:
            # make_primal
            self.q, self.qn = moto.sym.states(name + "_q", self.nq)
            if q_nom is not None:
                assert q_nom.shape == (self.nq,), "q_nom has wrong shape"
                self.q.sym_handle.default_value = q_nom
                self.qn.sym_handle.default_value = q_nom
            self.v, self.vn = moto.sym.states(name + "_v", self.nv)
            # self.aj = moto.sym.inputs(name + "_aj", self.nj)
            # self.a = moto.sym.inputs(name + "_a", self.nv)
            # self.aj = self.a[-self.nj :]
            # self.ab = (self.vn - self.v)[:6] / dt
            self.a = (self.vn - self.v) / dt
            # self.a = cs.vcat([self.ab, self.aj]) if self.is_floating_based else self.aj
            self.tq = moto.sym.inputs(name + "_tq", self.nj)

            # implicit euler
            def implicit_euler():
                q_next = cpin.integrate(self, self.q, self.vn * dt)
                # v_next = self.v[-self.nj :] + self.aj * dt
                # v_next = self.v + self.a * dt
                # return moto.dense_dynamics(
                #     name + "_euler",
                #     [self.q, self.v, self.qn, self.vn, self.a, dt],
                #     cs.vcat([self.qn - q_next, self.vn - v_next]),
                # )
                # return cs.vcat([self.qn - q_next, self.vn[-self.nj :] - v_next])
                # return cs.vcat([self.qn - q_next, self.vn - v_next])
                return cs.vcat([self.qn - q_next])

            self.joint_euler = implicit_euler()
            # self.ntq = model.nv - 6 if self.is_floating_based else model.nv
            self.q_stack = self.q
            self.v_stack = self.v
            self.qn_stack = self.qn
            self.vn_stack = self.vn
            self.a_stack = self.a
            self.pos_args = [self.q]
            self.vel_args = [self.v]
            # self.acc_args = [self.aj]
            # self.acc_args = [self.a]
            self.acc_args = [self.tq]
            self.pos_args_n = [self.qn]
            self.vel_args_n = [self.vn]
        else:
            self.euler = moto.euler_integrator(name + "_euler")
            self.euler.set_dt(dt)
            self.q_lin, self.qn_lin, self.v_lin, self.vn_lin, self.a_lin = self.euler.create_2nd_ord_lin(
                name, self.nv - (3 if self.is_floating_based else 0), semi_implicit=True
            )
            if q_nom is not None:
                assert q_nom.shape == (self.nq,), "q_nom has wrong shape"
                q_nom_lin = np.concatenate((q_nom[:3], q_nom[-self.nj :])) if self.is_floating_based else q_nom
                self.q_lin.default_value = q_nom_lin
                self.qn_lin.default_value = q_nom_lin
            if self.is_floating_based:
                self.quat, self.quatn, self.w, self.wn, self.dw = self.euler.create_2nd_ord_ang(
                    name, semi_implicit=True
                )
                if q_nom is not None:
                    self.quat.default_value = q_nom[3 : self.nqb]
                    self.quatn.default_value = q_nom[3 : self.nqb]
                else:
                    self.quat.default_value = np.array([0.0, 0.0, 0.0, 1.0])
                    self.quatn.default_value = np.array([0.0, 0.0, 0.0, 1.0])
                self.q_stack = cs.vcat([self.q_lin.to_sx()[:3], self.quat.to_sx(), self.q_lin.to_sx()[-self.nj :]])
                self.v_stack = cs.vcat([self.v_lin.to_sx()[:3], self.w.to_sx(), self.v_lin.to_sx()[-self.nj :]])
                self.qn_stack = cs.vcat([self.qn_lin.to_sx()[:3], self.quatn.to_sx(), self.qn_lin.to_sx()[-self.nj :]])
                self.vn_stack = cs.vcat([self.vn_lin.to_sx()[:3], self.wn.to_sx(), self.vn_lin.to_sx()[-self.nj :]])
                self.a_stack = cs.vcat([self.a_lin.to_sx()[:3], self.dw.to_sx(), self.a_lin.to_sx()[-self.nj :]])
                self.quat_args = [self.quat, self.quatn, self.w, self.wn, self.dw]

            self.pos_args = [self.q_lin] if not self.is_floating_based else [self.q_lin, self.quat]
            self.vel_args = [self.v_lin] if not self.is_floating_based else [self.v_lin, self.w]
            self.acc_args = [self.a_lin] if not self.is_floating_based else [self.a_lin, self.dw]
            self.pos_args_n = [self.qn_lin] if not self.is_floating_based else [self.qn_lin, self.quatn]
            self.vel_args_n = [self.vn_lin] if not self.is_floating_based else [self.vn_lin, self.wn]

        cpin.forwardKinematics(self, self.data, self.q_stack, self.v_stack)
        cpin.computeJointJacobians(self, self.data, self.q_stack)
        cpin.updateFramePlacements(self, self.data)

        if self.use_fwd_dyn:
            tau = cs.vcat([cs.SX.zeros(6), self.tq]) if self.is_floating_based else self.tq
            self.aba = cpin.aba(self, self.data, self.q_stack, self.v_stack, tau) * dt
        else:
            self.rnea = cpin.rnea(self, self.data, self.q_stack, self.v_stack, self.a_stack) * dt

        self.dyn = self.make_dynamics()

        self.q_nom = moto.sym.params("q_nom", self.nq, default_val=q_nom if q_nom is not None else np.zeros(self.nq))
        q_min = model.lowerPositionLimit[-self.nj :]
        q_max = model.upperPositionLimit[-self.nj :]
        v_lim = model.velocityLimit[-self.nj :]
        qj = self.q[-self.nj :]
        vj = self.v[-self.nj :]
        self.q_min = q_min
        self.q_max = q_max
        self.v_lim = v_lim

    def make_dynamics(self):
        args = self.pos_args + self.vel_args + self.pos_args_n + self.vel_args_n + self.acc_args
        if isinstance(self.dt, cs.SX):
            args.append(self.dt)
        if self.dense:
            out = [self.joint_euler]
            # out = [self.qn - cpin.integrate(self, self.q, self.vn * self.dt), self.vn - self.v - self.aba * self.dt]

            # if self.is_floating_based:
            # out.append(self.rnea_base)
            # return moto.dense_dynamics(self.name + "_fb_id", args, cs.vcat(out))
            in_arg = self.pos_args + self.vel_args + self.acc_args
            # return [moto.dense_dynamics(self.name + "_euler", args, cs.vcat(out)),
            #         moto.constr(self.name + "_id", in_arg, self.rnea[:6])]
            if self.use_fwd_dyn:
                v_next = self.v + self.aba
                return moto.dense_dynamics(self.name + "_fd", args, cs.vcat(out + [self.vn - v_next]))
            else:
                tau = cs.vcat([cs.SX.zeros(6), self.tq]) if self.is_floating_based else self.tq
                return moto.dense_dynamics(self.name + "_id", args, cs.vcat(out + [self.rnea - tau * self.dt]))
            # return moto.dense_dynamics(self.name + "_fb_fd", args, cs.vcat(out))
        else:
            to_add = [self.euler]
            if self.is_floating_based:
                args = self.pos_args + self.vel_args + self.acc_args + self.active_foot + self.f_f + [self.dt]
                to_add.append(moto.constr(self.name + "_fb_id", args, self.rnea_base))
            return to_add

    def make_ee_pos_constr(self):
        if not hasattr(self, "r_des"):
            self.r_des = moto.sym.params("r_des", 3, default_val=np.zeros(3))
            self.quat_des = moto.sym.params("quat_des", 4, default_val=np.array([0.0, 0.0, 0.0, 1.0]))
        ee_des = cpin.XYZQUATToSE3(cs.vcat([self.r_des, self.quat_des]))
        ee_pos = self.data.oMf[self.ee_id]
        # return moto.constr(
        #     "ee_constr", self.pos_args + [self.r_des, self.quat_des], cpin.log6(ee_pos.inverse() * ee_des).np
        # )
        res = cpin.log6(ee_des.inverse() * ee_pos).np
        # res = cs.vcat([ee_pos.translation - ee_des.translation, cpin.log3(ee_des.rotation.T @ ee_pos.rotation)])
        # return moto.constr("ee_constr_ineq", self.pos_args + [self.r_des, self.quat_des], cs.vcat([res, -res])).as_ineq()
        if not hasattr(self, "W_ee_cost"):
            self.W_ee_cost = moto.sym.params("W_ee_cost", 6, default_val=np.array([40.0, 40.0, 40.0, 0.0, 0.0, 0.0]))
        ee_lifted = moto.sym.inputs("ee_lifted", 6, default_val=np.zeros(6))
        return (
            moto.cost(
                # "ee_cost", self.pos_args + [self.r_des, self.quat_des, self.W_ee_cost], cs.sumsqr(self.W_ee_cost * res)
                "ee_cost",
                self.pos_args + [self.r_des, self.quat_des],
                res,
            )
            .set_gauss_newton(self.W_ee_cost)
            .as_terminal()
        )
        # return [
        #     moto.cost(
        #         "ee_cost_lifted", [ee_lifted, self.W_ee_cost], 0.5 * cs.sumsqr(cs.sqrt(self.W_ee_cost) * ee_lifted)
        #     ),  # .as_terminal(),
        #     moto.constr("ee_constr_lifted", self.pos_args + [self.r_des, self.quat_des, ee_lifted], ee_lifted - res),
        # ]

    def make_joint_limit_constr(self):
        q_min = self.fmodel.lowerPositionLimit[-self.nj :]
        q_max = self.fmodel.upperPositionLimit[-self.nj :]
        v_lim = self.fmodel.velocityLimit[-self.nj :]
        qj = self.q[-self.nj :]
        vj = self.v[-self.nj :]
        self.q_min = q_min
        self.q_max = q_max
        self.v_lim = v_lim
        return moto.constr(
            "q_limit", [self.q, self.v], cs.vcat([q_min - qj, qj - q_max, vj - v_lim, -vj - v_lim])
        ).as_ineq()

    def make_tq_limit_constr(self):
        tq_limit = model.effortLimit[-self.nj :]
        in_arg = [self.tq]
        # in_arg = self.pos_args + self.vel_args + self.acc_args + [*self.active_foot, *self.f_f] + ([self.dt] if isinstance(self.dt, cs.SX) else [])
        return moto.constr("tq_limit", in_arg, cs.vcat([self.tq - tq_limit, -self.tq - tq_limit])).as_ineq()

    def get_state_cost(self, terminal: bool = False):
        q_nom_res = self.q_stack - self.q_nom
        if self.is_floating_based:
            state_cost = (
                100.0 * cs.sumsqr(q_nom_res[: self.nqb])
                + 1 * cs.sumsqr(q_nom_res[self.nqb :])
                + 1.0 * cs.sumsqr(self.v_stack[:6])
                + 0.01 * cs.sumsqr(self.v_stack[6:])
            )
        else:
            state_cost = 0.1 * cs.sumsqr(q_nom_res) + 0.1 * cs.sumsqr(self.v_stack)
        state_args = self.pos_args + self.vel_args
        cost = moto.cost("c", state_args + [self.q_nom], state_cost).set_diag_hess()
        if terminal:
            return cost.as_terminal()
        return cost

    def get_input_cost(self):
        input_args = self.acc_args
        if self.dense:
            input_cost = 1e-4 * cs.sumsqr(self.tq)
        else:
            input_cost = 1e-4 * cs.sumsqr(self.a_stack)
        return moto.cost("c_u", input_args, input_cost).set_diag_hess()


# dt = moto.sym.inputs("dt", 1, default_val=0.02)
dt_nom = moto.sym.params("dt_nom", 1, default_val=0.02)
dt = 0.02
display = True
ur5 = load("ur5_limited", display=display, verbose=True)
q_d = np.copy(ur5.q0)
# q_d = np.array([0.0, -np.pi / 2, 0.0, -np.pi / 2, 0.0, 0.0])
model = pin.buildModelFromUrdf(ur5.urdf)
# model.gravity.linear = np.array([0.0, 0.0, 0])
np.set_printoptions(precision=3, suppress=True, linewidth=200)
model = pinCasadiModel(model, dt=dt, q_nom=q_d, dense=True, use_fwd_dyn=False)

prob = moto.ocp.create()
prob.add(model.dyn)
prob.add(model.make_tq_limit_constr())
prob.add(model.make_joint_limit_constr())
prob.add(model.get_state_cost())
prob.add(model.get_input_cost())

prob_term = prob.clone()
prob_term.add(model.make_ee_pos_constr())
prob_term.add(model.get_state_cost(terminal=True))

prob.print_summary()
prob_term.print_summary()
print("--" * 15)
# moto.print_problem(prob_term)

N_horizon = 50

sqp = moto.sqp(n_job=1)
g = sqp.graph
n0 = g.set_head(g.add(sqp.create_node(prob)))
n1 = g.set_tail(g.add(sqp.create_node(prob_term)))
g.add_edge(n0, n1, N_horizon)
cfg = [
    [
        -0.4413381175136468,
        -0.3681474968347521,
        0.5048906263414011,
        -0.3459516731344637,
        -0.012699514214883486,
        0.9256556113790925,
        0.1527018379606805,
    ],
    [
        -0.39183453679515345,
        2.7132996805605147,
        2.7068577763863484,
        1.3882845155376211,
        -1.3553245206176137,
        1.498755302649343,
    ],
]
cfg = [
    [0.4, 0.4, 0.4, 0.0, 0.0, 0.0, 1.0],
    [0.0] * 6,
]

n1.data.value[model.r_des] = np.array(cfg[0][:3])
n1.data.value[model.quat_des] = np.array(cfg[0][3:7])
n1.data.value[model.W_ee_cost] = np.ones(6) * 1e6


def set_initial_state(data: moto.sqp.data_type):
    data.value[model.q] = np.array(cfg[1])
    data.value[model.qn] = np.array(cfg[1])
    # data.value[model.r_des] = np.array(cfg[0][:3])
    # data.value[model.quat_des] = np.array(cfg[0][3:7])
    # data.value[model.W_ee_cost] = np.ones(6) * 4


sqp.apply_forward(set_initial_state)

sqp.settings.mu = 1
# sqp.settings.mu_method = moto.sqp.adaptive_mu_t.mehrotra_probing
sqp.settings.mu_method = moto.sqp.adaptive_mu_t.mehrotra_predictor_corrector
# sqp.settings.mu_method = moto.sqp.adaptive_mu_t.quality_function_based
sqp.settings.ipm_conditional_corrector = True
sqp.settings.prim_tol = 1e-3
sqp.settings.dual_tol = 1e-3
sqp.settings.comp_tol = 1e-3

import time

start = time.perf_counter()
sqp.update(100)
print(f"sqp.update(100) took {time.perf_counter() - start:.3f} seconds")

q_res = []
dt_res = []
node_idx = 0


def get_sym(node: moto.sqp.data_type):
    global node_idx
    q_res.append(node.value[model.q])
    if isinstance(dt, float):
        dt_res.append(dt)
    else:
        dt_res.append(node.value[dt])
    node_idx += 1
    if node_idx >= N_horizon:
        q_res.append(node.value[model.qn])


sqp.apply_forward(get_sym)

# compute the final end-effector position error
fdata = model.fmodel.createData()
pin.forwardKinematics(model.fmodel, fdata, q_res[-1])
pin.updateFramePlacements(model.fmodel, fdata)
eef = fdata.oMf[model.ee_id]
eef_des = pin.XYZQUATToSE3(cfg[0])
print("final ee pos err:", pin.log6(eef_des.inverse() * eef).np)

if display:

    import time

    data = model.fmodel.createData()
    while True:
        for i in range(len(q_res)):
            # print((q_res[i] - model.q_max) > 0)
            # print((model.q_min - q_res[i]) > 0)
            start = time.perf_counter()
            ur5.display(q_res[i])
            if i is not N_horizon:
                dt_ = dt_res[i]
                while time.perf_counter() - start < dt_:
                    pass
        time.sleep(0.5)

# def print_sym(node: moto.sqp.data_type):
#     node.sym.print()
#     node.print_residuals()

# sqp.apply_forward(print_sym, early_stop=20)
