import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

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

        # make_primal
        self.q, self.qn = moto.sym.states(name + "_q", self.nq)
        if q_nom is not None:
            assert q_nom.shape == (self.nq,), "q_nom has wrong shape"
            self.q.default_value = q_nom
            self.qn.default_value = q_nom
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
            q_next = cpin.integrate(self, self.q.sx, self.vn * dt)
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
        self.q_stack = self.q.sx
        self.v_stack = self.v.sx
        self.qn_stack = self.qn.sx
        self.vn_stack = self.vn.sx
        self.a_stack = self.a
        self.pos_args = [self.q]
        self.vel_args = [self.v]
        # self.acc_args = [self.aj]
        # self.acc_args = [self.a]
        self.acc_args = [self.tq]
        self.pos_args_n = [self.qn]
        self.vel_args_n = [self.vn]

        cpin.forwardKinematics(self, self.data, self.q_stack, self.v_stack)
        cpin.computeJointJacobians(self, self.data, self.q_stack)
        cpin.updateFramePlacements(self, self.data)

        if self.use_fwd_dyn:
            tau = (
                cs.vcat([cs.SX.zeros(6), self.tq])
                if self.is_floating_based
                else self.tq
            )
            self.aba = cpin.aba(self, self.data, self.q_stack, self.v_stack, tau) * dt
        else:
            self.rnea = (
                cpin.rnea(self, self.data, self.q_stack, self.v_stack, self.a_stack)
                * dt
            )

        self.dyn = self.make_dynamics()

        self.q_nom = moto.sym.params(
            "q_nom",
            self.nq,
            default_val=q_nom if q_nom is not None else np.zeros(self.nq),
        )
        q_min = model.lowerPositionLimit[-self.nj :]
        q_max = model.upperPositionLimit[-self.nj :]
        v_lim = model.velocityLimit[-self.nj :]
        self.q_min = q_min
        self.q_max = q_max
        self.v_lim = v_lim

    def make_dynamics(self):
        args = (
            self.pos_args
            + self.vel_args
            + self.pos_args_n
            + self.vel_args_n
            + self.acc_args
        )
        if isinstance(self.dt, cs.SX):
            args.append(self.dt)
        out = [self.joint_euler]
        # out = [self.qn - cpin.integrate(self, self.q, self.vn * self.dt), self.vn - self.v - self.aba * self.dt]

        # if self.is_floating_based:
        # out.append(self.rnea_base)
        # return moto.dense_dynamics(self.name + "_fb_id", args, cs.vcat(out))
        in_arg = self.pos_args + self.vel_args + self.acc_args
        # return [moto.dense_dynamics(self.name + "_euler", args, cs.vcat(out)),
        #         moto.constr.create(self.name + "_id", in_arg, self.rnea[:6])]
        if self.use_fwd_dyn:
            v_next = self.v + self.aba
            return moto.dense_dynamics.create(
                "arm_" + self.name + "_fd", args, cs.vcat(out + [self.vn - v_next])
            )
        else:
            tau = (
                cs.vcat([cs.SX.zeros(6), self.tq])
                if self.is_floating_based
                else self.tq
            )
            return moto.dense_dynamics.create(
                "arm_" + self.name + "_id",
                args,
                cs.vcat(out + [self.rnea - tau * self.dt]),
            )
        # return moto.dense_dynamics(self.name + "_fb_fd", args, cs.vcat(out))

    def make_ee_pos_constr(self):
        if not hasattr(self, "r_des"):
            self.r_des = moto.sym.params("r_des", 3, default_val=np.zeros(3))
            self.quat_des = moto.sym.params(
                "quat_des", 4, default_val=np.array([0.0, 0.0, 0.0, 1.0])
            )
        ee_des = cpin.XYZQUATToSE3(cs.vcat([self.r_des, self.quat_des]))
        ee_pos = self.data.oMf[self.ee_id]
        c: moto.pmm_constr = moto.constr.create(
            "arm_ee_constr",
            self.pos_args + [self.r_des, self.quat_des],
            cpin.log6(ee_pos.inverse() * ee_des).np,
        )
        return c
        c = c.cast_soft()
        c.rho = 1e-4
        return c
        res = cpin.log6(ee_des.inverse() * ee_pos).np
        # res = cs.vcat([ee_pos.translation - ee_des.translation, cpin.log3(ee_des.rotation.T @ ee_pos.rotation)])
        # return moto.ineq.create("ee_constr_ineq", self.pos_args + [self.r_des, self.quat_des], cs.vcat([res, -res]))
        if not hasattr(self, "W_ee_cost"):
            self.W_ee_cost = moto.sym.params(
                "W_ee_cost", 6, default_val=np.array([40.0, 40.0, 40.0, 0.0, 0.0, 0.0])
            )
        ee_lifted = moto.sym.inputs("ee_lifted", 6, default_val=np.zeros(6))
        return moto.cost.create(
            # "arm_ee_cost", self.pos_args + [self.r_des, self.quat_des, self.W_ee_cost], cs.sumsqr(self.W_ee_cost * res)
            "arm_ee_cost",
            self.pos_args + [self.r_des, self.quat_des],
            res,
        ).set_gauss_newton(self.W_ee_cost)
        # return [
        #     moto.cost.create(
        #         "ee_cost_lifted", [ee_lifted, self.W_ee_cost], 0.5 * cs.sumsqr(cs.sqrt(self.W_ee_cost) * ee_lifted)
        #     ),  # .as_terminal(),
        #     moto.constr.create("ee_constr_lifted", self.pos_args + [self.r_des, self.quat_des, ee_lifted], ee_lifted - res),
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
        return moto.ineq.create(
            "arm_q_limit",
            [self.q, self.v],
            cs.vcat([qj, vj]),
            np.concatenate([q_min, -v_lim]),
            np.concatenate([q_max, v_lim]),
        )

    def make_tq_limit_constr(self):
        tq_limit = self.fmodel.effortLimit[-self.nj :]
        in_arg = [self.tq]
        # in_arg = self.pos_args + self.vel_args + self.acc_args + [*self.active_foot, *self.f_f] + ([self.dt] if isinstance(self.dt, cs.SX) else [])
        return moto.ineq.create(
            "arm_tq_limit", in_arg, self.tq.sx, -tq_limit, tq_limit
        )

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
        cost = moto.cost.create(
            "arm_state_cost", state_args + [self.q_nom], state_cost
        ).set_diag_hess()
        if terminal:
            return cost
        return cost

    def get_input_cost(self):
        input_args = self.acc_args
        input_cost = 1e-4 * cs.sumsqr(self.tq)
        return moto.cost.create(
            "arm_input_cost", input_args, input_cost
        ).set_diag_hess()

    def get_dt_reg(self, dt_nom):
        if not isinstance(self.dt, cs.SX):
            raise ValueError("dt is not a symbolic variable")
        W_dt = moto.sym.params("W_dt", 1, default_val=1e3)
        return [
            moto.cost.create(
                "arm_dt_reg", [self.dt, dt_nom, W_dt], W_dt * (self.dt - dt_nom) ** 2
            ),
            moto.ineq.create("arm_dt_bound", [self.dt], self.dt.sx, 1e-2, 0.1),
        ]


# dt = moto.sym.inputs("dt", 1, default_val=0.02)
dt_nom = moto.sym.params("dt_nom", 1, default_val=0.02)
dt = 0.02


def build_sqp(display: bool, n_job: int = 4):
    ur5 = load("ur5_limited", display=display, verbose=True)
    q_d = np.copy(ur5.q0)
    model = pin.buildModelFromUrdf(ur5.urdf)
    np.set_printoptions(precision=3, suppress=True, linewidth=200)
    model = pinCasadiModel(model, dt=dt, q_nom=q_d, dense=True, use_fwd_dyn=False)

    stage_prob = moto.node_ocp.create()
    stage_prob.add(model.make_tq_limit_constr())
    stage_prob.add(model.make_joint_limit_constr())
    stage_prob.add(model.get_state_cost())
    stage_prob.add(model.get_input_cost())

    terminal_prob = stage_prob.clone()
    terminal_prob.add_terminal(model.make_ee_pos_constr())

    stage_prob.print_summary()
    terminal_prob.print_summary()
    print("--" * 15)

    horizon = 50
    sqp = moto.sqp(n_job=n_job)
    modeled = sqp.create_graph()
    start_node = modeled.create_node(stage_prob)
    terminal_node = modeled.create_node(terminal_prob)
    for edge in modeled.add_path(start_node, terminal_node, horizon):
        edge.add(model.dyn)

    # cfg = [
    #     [
    #         0.29330284554440844,
    #         -0.4822177449570157,
    #         0.35490485263950977,
    #         0.685024579972181,
    #         -0.4567935245973314,
    #         -0.2681559031137524,
    #         -0.5001733822837158,
    #     ],
    #     [
    #         0.3602555199390869,
    #         -0.9113326884740325,
    #         -0.28553691838581385,
    #         0.36474866340717416,
    #         0.06126558758673073,
    #         0.8544954685368089,
    #     ],
    # ]
    # bad setting
    cfg = [
        [
            -0.20386130721377144,
            -0.2780080314109077,
            0.05367065178434891,
            0.6564066614217499,
            -0.7069685526302121,
            0.262510077975368,
            -0.020352380560065237,
        ],
        [
            0.39077402479947176,
            0.2908006855904619,
            -0.5419511019201957,
            0.22346096937115068,
            -0.20115318796212112,
            0.3804248223817164,
        ],
    ]

    def set_initial_state(data: moto.sqp.data_type):
        data.value[model.q] = np.array(cfg[1])
        data.value[model.qn] = np.array(cfg[1])
        if data.prob.dim(moto.field___eq_x) > 0:
            data.value[model.r_des] = np.array(cfg[0][:3])
            data.value[model.quat_des] = np.array(cfg[0][3:7])
            if hasattr(model, "W_ee_cost"):
                data.value[model.W_ee_cost] = np.ones(6) * 1e8

    sqp.apply_forward(set_initial_state)

    sqp.settings.ipm.mu0 = 0.1
    # sqp.settings.ipm.mu_method = moto.sqp.adaptive_mu_t.mehrotra_predictor_corrector
    sqp.settings.ipm.mu_method = moto.sqp.monotonic_decrease
    sqp.settings.prim_tol = 1e-3
    sqp.settings.dual_tol = 1e-3
    sqp.settings.comp_tol = 1e-3
    sqp.settings.restoration.rho_eq = 1e3
    sqp.settings.restoration.rho_ineq = 1e3
    sqp.settings.restoration.rho_u = 0.0001
    sqp.settings.restoration.rho_y = 0.0001
    sqp.settings.ls.update_alpha_dual = False

    return sqp, model, ur5, cfg, horizon


def collect_trajectory(sqp, model, horizon):
    q_res = []
    dt_res = []
    node_idx = 0

    def get_sym(node: moto.sqp.data_type):
        nonlocal node_idx
        q_res.append(node.value[model.q])
        if isinstance(dt, float):
            dt_res.append(dt)
        else:
            dt_res.append(node.value[dt])
        node_idx += 1
        if node_idx >= horizon:
            q_res.append(node.value[model.qn])

    sqp.apply_forward(get_sym)
    return q_res, dt_res


def visualize_solution(ur5, model, cfg, q_res, dt_res, horizon):
    import meshcat.geometry as mg
    import meshcat.transformations as tf
    import meshcat_shapes as mcs
    import time
    from scipy.spatial.transform.rotation import Rotation as R

    viz = ur5.viz
    target = mg.Sphere(0.02)
    viz.viewer["/target"].set_object(target)
    quat = cfg[0][3:7]
    r = R.from_quat(quat).as_matrix()
    target_pose = tf.compose_matrix(translate=cfg[0][:3])
    rot = np.eye(4)
    rot[:3, :3] = r
    viz.viewer["/target"].set_transform(target_pose)

    mcs.frame(viz.viewer["/frame"])
    viz.viewer["/frame"].set_transform(target_pose.dot(rot))

    while True:
        for i in range(len(q_res)):
            start = time.perf_counter()
            ur5.display(q_res[i])
            if i != horizon:
                dt_ = dt_res[i]
                while time.perf_counter() - start < dt_:
                    pass
        time.sleep(0.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--display", action="store_true", help="enable meshcat visualization"
    )
    parser.add_argument(
        "--n-job", type=int, default=4, help="number of solver worker jobs"
    )
    parser.add_argument(
        "--max-iter", type=int, default=100, help="maximum SQP iterations"
    )
    args = parser.parse_args()

    sqp, model, ur5, cfg, horizon = build_sqp(display=args.display, n_job=args.n_job)

    import time

    start = time.perf_counter()
    kkt = sqp.update(args.max_iter)
    print(f"sqp.update({args.max_iter}) took {time.perf_counter() - start:.3f} seconds")
    print(f"result       : {kkt.result}")
    print(f"num_iter     : {kkt.num_iter}")
    print(f"prim_res     : {kkt.inf_prim_res:.3e}")
    print(f"dual_res     : {kkt.inf_dual_res:.3e}")
    print(f"comp_res     : {kkt.inf_comp_res:.3e}")
    print(f"solved       : {kkt.solved}")

    q_res, dt_res = collect_trajectory(sqp, model, horizon)

    fdata = model.fmodel.createData()
    pin.forwardKinematics(model.fmodel, fdata, q_res[-1])
    pin.updateFramePlacements(model.fmodel, fdata)
    eef = fdata.oMf[model.ee_id]
    eef_des = pin.XYZQUATToSE3(cfg[0])
    print("final ee pos err:", pin.log6(eef_des.inverse() * eef).np)

    if args.display:
        visualize_solution(ur5, model, cfg, q_res, dt_res, horizon)


if __name__ == "__main__":
    main()

# def print_sym(node: moto.sqp.data_type):
#     node.sym.print()
#     node.print_residuals()

# sqp.apply_forward(print_sym, early_stop=20)
