"""Shared Pinocchio helpers used by the robot examples.

The helpers in this module keep Pinocchio's model/data plumbing and the
discrete impulse conventions out of the individual OCP examples.
"""

from __future__ import annotations

import casadi as cs
import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin

import moto

GO2_FOOT_FRAMES = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")


def build_floating_base_model(urdf, orientation="quaternion"):
    """Build the common translation plus quaternion/ZYX root joint."""
    root_joint = pin.JointModelComposite()
    root_joint.addJoint(pin.JointModelTranslation())
    if orientation == "quaternion":
        root_joint.addJoint(pin.JointModelSpherical())
    elif orientation == "euler_zyx":
        root_joint.addJoint(pin.JointModelSphericalZYX())
    else:
        raise ValueError("orientation must be 'quaternion' or 'euler_zyx'")
    return pin.buildModelFromUrdf(urdf, root_joint)


def pinocchio_states(model, name, default=None):
    """Create q/q-next manifold states and matching v/v-next states."""
    if default is not None:
        default = np.asarray(default)
        if default.shape != (model.nq,):
            raise ValueError(f"default must have shape ({model.nq},)")
    q_name = f"{name}_q"
    q0 = cs.SX.sym(f"{q_name}_base", model.nq)
    q1 = cs.SX.sym(f"{q_name}_other", model.nq)
    dq = cs.SX.sym(f"{q_name}_step", model.nv)
    args = (
        q_name,
        q0,
        dq,
        cpin.integrate(model, q0, dq),
        q1,
        cpin.difference(model, q0, q1),
    )
    q, qn = (
        moto.casadi_manifold.create(*args)
        if default is None
        else moto.casadi_manifold.create(*args, default)
    )
    v, vn = moto.sym.states(f"{name}_v", model.nv)
    return q, qn, v, vn


def semi_implicit_dynamics(name, q, v, qn, vn, velocity_residual, dt):
    """Create manifold semi-implicit Euler dynamics without exposing Pinocchio ops."""
    q_next = q.symbolic_integrate(q.sx, vn * dt)
    residual = cs.vcat([q.symbolic_difference(qn.sx, q_next), velocity_residual])
    return moto.sparse_dynamics.create(name, residual)


class PinocchioCasadiModel(cpin.Model):
    """CasADi Pinocchio model with common kinematics/dynamics operations."""

    def __init__(self, model: pin.Model, name: str = "") -> None:
        self.fmodel = model
        super().__init__(model)

        self.name = name or model.name.replace("_description", "")
        self.data = self.createData()
        self.is_floating_based = self.nv > self.njoints
        self.nj = self.nv - 6 if self.is_floating_based else self.nv
        self.nqb = 7 if self.nq - self.nv == 1 else 6 if self.is_floating_based else 0

    def create_trajectory_variables(self, dt, q_nom=None):
        """Create the standard q/v state pair and actuated torque input."""
        self.dt = dt
        self.q, self.qn, self.v, self.vn = pinocchio_states(
            self, self.name, q_nom
        )
        self.a = (self.vn - self.v) / dt
        self.tq = moto.sym.inputs(f"{self.name}_tq", self.nj)

        self.q_stack = self.q.sx
        self.v_stack = self.v.sx
        self.qn_stack = self.qn.sx
        self.vn_stack = self.vn.sx
        self.a_stack = self.a
        return self

    def update_kinematics(self, q, v=None, *, data=None, jacobians=True):
        """Update joint/frame placements and, optionally, joint Jacobians."""
        data = self.data if data is None else data
        if v is None:
            cpin.forwardKinematics(self, data, q)
        else:
            cpin.forwardKinematics(self, data, q, v)
        if jacobians:
            cpin.computeJointJacobians(self, data, q)
        cpin.updateFramePlacements(self, data)
        return data

    def frame_translational_jacobians(self, frame_ids, *, data=None):
        data = self.data if data is None else data
        return [
            cpin.getFrameJacobian(self, data, frame_id, pin.LOCAL_WORLD_ALIGNED)[:3, :]
            for frame_id in frame_ids
        ]

    def frame_linear_kinematics(self, frame_ids, q, v):
        """Return frame linear velocities (columns) and frame heights."""
        data = self.update_kinematics(q, v, data=self.createData())
        velocities = cs.hcat(
            [
                cpin.getFrameVelocity(
                    self, data, frame_id, pin.LOCAL_WORLD_ALIGNED
                ).linear
                for frame_id in frame_ids
            ]
        )
        heights = cs.vcat([data.oMf[frame_id].translation[2] for frame_id in frame_ids])
        return velocities, heights

    def generalized_torque(self, joint_torque):
        """Expand actuated joint torque to the model's generalized tangent space."""
        if self.is_floating_based:
            return cs.vcat([cs.SX.zeros(6), joint_torque])
        return joint_torque

    def forward_dynamics_step(self, q, v, joint_torque, dt, external_impulse=0):
        """Compute ``dv = a * dt`` using ABA and an external impulse."""
        torque = self.generalized_torque(joint_torque) + external_impulse / dt
        return cpin.aba(self, self.data, q, v, torque) * dt

    def inverse_dynamics_impulse(self, q, v, a, dt, external_impulse=0):
        """Compute the RNEA generalized impulse minus an external impulse."""
        return cpin.rnea(self, self.data, q, v, a) * dt - external_impulse

    def joint_limit_constraint(self, q, v, *, name="q_limit"):
        """Create the standard actuated joint position/velocity box constraint."""
        q_joint = q[-self.nj :]
        v_joint = v[-self.nj :]
        q_min = self.fmodel.lowerPositionLimit[-self.nj :]
        q_max = self.fmodel.upperPositionLimit[-self.nj :]
        v_limit = self.fmodel.velocityLimit[-self.nj :]
        return moto.ineq.create(
            name,
            cs.vcat([q_joint, v_joint]),
            np.concatenate([q_min, -v_limit]),
            np.concatenate([q_max, v_limit]),
        )

    def torque_limit_constraint(self, torque, *, name="tq_limit"):
        """Create the standard actuated joint torque box constraint."""
        limit = self.fmodel.effortLimit[-self.nj :]
        return moto.ineq.bounds(name, torque, -limit, limit)


class ContactModel:
    """Shared symbolic point-contact model for the robot examples.

    The contact variables represent impulses over one integration interval.
    ``generalized_impulse`` can therefore be passed directly to
    :meth:`PinocchioCasadiModel.forward_dynamics_step`.
    """

    def __init__(
        self,
        robot: PinocchioCasadiModel,
        q,
        v,
        frame_names,
        *,
        force_prefix="f",
        force_default=(0.0, 0.0, 0.5),
        kinematic_gain=100.0,
        friction_coefficient=0.7,
    ):
        self.robot = robot
        self.q = q
        self.v = v
        self.frame_names = tuple(frame_names)
        self.frame_ids = tuple(robot.fmodel.getFrameId(name) for name in frame_names)

        robot.update_kinematics(q.sx, v.sx)
        self.jacobians = robot.frame_translational_jacobians(self.frame_ids)
        self.impulses = [
            moto.sym.inputs(
                f"{force_prefix}_{name}", 3, default_val=np.asarray(force_default)
            )
            for name in self.frame_names
        ]
        self.generalized_impulses = [
            jacobian.T @ force for jacobian, force in zip(self.jacobians, self.impulses)
        ]
        self.generalized_impulse = sum(self.generalized_impulses, cs.SX.zeros(robot.nv))
        self.velocities, self.heights = robot.frame_linear_kinematics(
            self.frame_ids, q.sx, v.sx
        )

        self.kinematic_gain = moto.sym.params("k_f", default_val=kinematic_gain)
        self.friction_coefficient = moto.sym.params(
            "mu", default_val=friction_coefficient
        )
        self.kinematic_constraints = [
            self._make_kinematic_constraint(index)
            for index in range(len(self.frame_names))
        ]
        self.friction_constraints = [
            self._make_friction_constraint(index)
            for index in range(len(self.frame_names))
        ]

    def _make_kinematic_constraint(self, index):
        velocity = self.velocities[:, index]
        residual = cs.vcat(
            [
                velocity[:2],
                self.kinematic_gain * self.heights[index] + velocity[2],
            ]
        )
        constraint = moto.constr.create(
            f"kin_{self.frame_names[index]}",
            residual,
        )
        constraint.enable_if_all([self.impulses[index]])
        return constraint

    def _make_friction_constraint(self, index):
        force = self.impulses[index]
        mu = self.friction_coefficient
        residual = cs.vcat(
            [
                force[0] - mu * force[2],
                -force[0] - mu * force[2],
                force[1] - mu * force[2],
                -force[1] - mu * force[2],
            ]
        )
        constraint = moto.ineq.create(f"fric_{self.frame_names[index]}", residual)
        constraint.enable_if_all([force])
        return constraint

    def add_to_stage(self, stage):
        """Add interval friction and start-node contact kinematics."""
        stage.add(self.friction_constraints)
        stage.st.add(self.kinematic_constraints)

    def add_to_endpoint(self, endpoint):
        endpoint.add(self.kinematic_constraints)

    def set_kinematic_gain(self, nodes, value, *, count=None):
        set_node_value(nodes, self.kinematic_gain, value, count=count)


class ContactRobotModel(PinocchioCasadiModel):
    """Ready-to-use q/v/torque/contact dynamics model for OCP examples."""

    def __init__(
        self,
        model: pin.Model,
        *,
        name="",
        dt=0.01,
        q_nom=None,
        contact_frames=GO2_FOOT_FRAMES,
        use_forward_dynamics=True,
        configuration_velocity="next",
    ):
        super().__init__(model, name)
        self.use_fwd_dyn = use_forward_dynamics
        self.create_trajectory_variables(dt, q_nom)
        self.contacts = ContactModel(self, self.q, self.v, contact_frames)

        if use_forward_dynamics:
            self.aba = self.forward_dynamics_step(
                self.q_stack,
                self.v_stack,
                self.tq,
                dt,
                self.contacts.generalized_impulse,
            )
        else:
            self.rnea = self.inverse_dynamics_impulse(
                self.q_stack,
                self.v_stack,
                self.a_stack,
                dt,
                self.contacts.generalized_impulse,
            )

        if configuration_velocity not in ("next", "predicted"):
            raise ValueError("configuration_velocity must be 'next' or 'predicted'")
        self.configuration_velocity = configuration_velocity
        self.dyn = self._make_contact_dynamics()

        self.q_nom = moto.sym.params(
            "q_nom",
            self.nq,
            default_val=q_nom if q_nom is not None else np.zeros(self.nq),
        )

    def _make_contact_dynamics(self):
        if self.use_fwd_dyn:
            velocity_residual = self.vn - (self.v + self.aba)
            name = f"{self.name}_fd"
        else:
            velocity_residual = self.rnea - self.generalized_torque(self.tq) * self.dt
            name = f"{self.name}_id"
        if self.configuration_velocity == "next":
            return semi_implicit_dynamics(
                name, self.q, self.v, self.qn, self.vn, velocity_residual, self.dt
            )
        integration_velocity = self.v + self.aba if self.use_fwd_dyn else self.vn
        q_next = self.q.symbolic_integrate(self.q.sx, integration_velocity * self.dt)
        return moto.sparse_dynamics.create(
            name, cs.vcat([self.q.symbolic_difference(self.qn.sx, q_next), velocity_residual])
        )

    def input_cost(self, *, torque_weight=1e-6, contact_weight=1e-3, name="c_u"):
        impulses = cs.vcat(self.contacts.impulses)
        residual = cs.vcat([self.tq, impulses])
        weight = np.r_[
            np.full(self.tq.numel(), 2 * torque_weight),
            np.full(impulses.numel(), 2 * contact_weight),
        ]
        return moto.cost.from_vector(name, residual, weight=weight)


def solver_nodes(sqp):
    """Materialize solver nodes once for reuse by an example."""
    return list(sqp.flatten_nodes())


def add_terms(container, *terms):
    """Add several expressions or expression lists to an OCP container."""
    for term in terms:
        container.add(term)
    return container


def add_stage_segments(sqp, stage_prototypes, lengths):
    """Append several linear graph segments and return all owned stages."""
    stage_prototypes = tuple(stage_prototypes)
    lengths = tuple(lengths)
    if len(stage_prototypes) != len(lengths):
        raise ValueError("stage_prototypes and lengths must have the same size")
    phases = list(zip(stage_prototypes, lengths))
    return [stage for phase in sqp.add_phases(phases) for stage in phase]


def add_time_step_regularization(
    stage,
    dt,
    nominal_dt,
    *,
    lower=1e-4,
    upper=5e-2,
    weight=1e8,
):
    """Add the shared symbolic time-step bound and quadratic regularization."""
    if not isinstance(dt, cs.SX):
        return
    bounds = moto.sym.params("dt_bound", 2, default_val=np.array([lower, upper]))
    add_terms(
        stage,
        moto.ineq.create("dt", dt.sx, bounds[0], bounds[1]),
        moto.cost.from_scalar(
            "c_t", dt - nominal_dt, weight=2 * weight
        ),
    )


def visit_nodes(nodes, callback):
    """Apply ``callback(node, index)`` to each already-materialized node."""
    for index, node in enumerate(nodes):
        callback(node, index)
    return nodes


def set_node_value(nodes, symbol, value, *, count=None):
    """Assign one symbol across all or the first ``count`` solver nodes."""
    selected = nodes if count is None else nodes[:count]
    for node in selected:
        node.value[symbol] = value
    return nodes


def collect_node_values(nodes, *symbols):
    """Collect copied numeric values for one or more symbols at every node."""
    return tuple(
        [np.array(node.value[symbol], copy=True) for node in nodes]
        for symbol in symbols
    )


def collect_state_trajectory(nodes, state, next_state, dt):
    """Collect interval states plus the final outgoing state and time steps."""
    states = [np.array(node.value[state], copy=True) for node in nodes]
    states.append(np.array(nodes[-1].value[next_state], copy=True))
    if isinstance(dt, (float, int)):
        time_steps = [float(dt)] * len(nodes)
    else:
        time_steps = [float(node.value[dt]) for node in nodes]
    return states, time_steps


def print_graph_layout(nodes):
    """Print the compact x/u/y layout of materialized solver nodes."""
    for index, node in enumerate(nodes):
        prob = node.prob
        print(
            f"  node[{index}] "
            f"x={prob.dim(moto.field.field___x)} "
            f"u={prob.dim(moto.field.field___u)} "
            f"y={prob.dim(moto.field.field___y)}"
        )


def frame_placement(model: pin.Model, q, frame_id: int):
    """Evaluate one frame placement on a numeric Pinocchio model."""
    data = model.createData()
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    return data.oMf[frame_id]


def numeric_frame_linear_kinematics(model, data, q, v, frame_ids):
    """Evaluate frame heights and linear velocities on a numeric model."""
    pin.forwardKinematics(model, data, q, v)
    pin.updateFramePlacements(model, data)
    heights = [data.oMf[frame_id].translation[2] for frame_id in frame_ids]
    velocities = [
        pin.getFrameVelocity(model, data, frame_id, pin.LOCAL_WORLD_ALIGNED).linear
        for frame_id in frame_ids
    ]
    return heights, velocities
