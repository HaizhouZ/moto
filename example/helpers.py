"""Shared Pinocchio helpers used by the robot examples.

The helpers in this module keep Pinocchio's model/data plumbing and the
discrete impulse conventions out of the individual OCP examples.
"""

from __future__ import annotations

from pathlib import Path

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
    return moto.semi_implicit_euler.create(name, residual)


def tangent_jacobian(output, argument):
    """Differentiate an SX expression in a moto symbol's tangent coordinates."""
    jacobian = cs.jacobian(output, argument.sx)
    if argument.dim == argument.tdim:
        return jacobian
    step = cs.SX.sym(f"{argument.name}_analytic_step", argument.tdim)
    tangent_map = cs.jacobian(argument.symbolic_integrate(argument.sx, step), step)
    tangent_map = cs.substitute(tangent_map, step, cs.SX.zeros(argument.tdim))
    return jacobian @ tangent_map


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

    def create_trajectory_variables(
        self,
        dt,
        q_nom=None,
        *,
        acceleration_control=False,
        lifted_contact=False,
        lifted_acceleration=False,
    ):
        """Create q/v states and either acceleration or torque input."""
        self.dt = dt
        self.q, self.qn, self.v, self.vn = pinocchio_states(self, self.name, q_nom)
        self.a = (
            moto.sym.inputs(f"{self.name}_a", self.nv)
            if acceleration_control
            else (
                moto.sym.lifted(f"{self.name}_a", self.nv)
                if lifted_contact and lifted_acceleration
                else (self.vn - self.v) / dt
            )
        )
        self.tq = (
            None
            if acceleration_control
            else moto.sym.inputs(f"{self.name}_tq", self.nj)
        )

        self.q_stack = self.q.sx
        self.v_stack = self.v.sx
        self.qn_stack = self.qn.sx
        self.vn_stack = self.vn.sx
        self.a_stack = self.a.sx if isinstance(self.a, moto.var) else self.a
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

    def forward_dynamics_step_analytic(
        self,
        q,
        v,
        joint_torque,
        dt,
        external_impulse=0,
        *,
        q_symbol,
        v_symbol,
        derivative_arguments=(),
    ):
        """Compute an ABA step and analytic tangent Jacobians.

        Pinocchio supplies the fixed-torque ABA derivatives. Derivatives of
        generalized torque, including ``J(q).T @ impulse``, are added through
        the chain rule for every requested moto symbol.
        """
        torque = self.generalized_torque(joint_torque) + external_impulse / dt
        dv = cpin.aba(self, self.data, q, v, torque) * dt
        derivative_data = self.createData()
        ddq_dq, ddq_dv, ddq_dtau = cpin.computeABADerivatives(
            self, derivative_data, q, v, torque
        )

        # Pinocchio's analytic ABA derivatives for a composite floating-base
        # joint are exact for all articulated-joint columns, but its first six
        # q/v columns use the wrong composite-joint convention. Keep the
        # compact analytic result and repair only those root columns with AD.
        if (
            self.is_floating_based
            and self.joints[1].shortname() == "JointModelComposite"
        ):
            root_tdim = self.joints[1].nv
            torque_q = tangent_jacobian(torque, q_symbol)
            ddq_dq[:, :root_tdim] = (
                tangent_jacobian(dv, q_symbol)[:, :root_tdim] / dt
                - (ddq_dtau @ torque_q)[:, :root_tdim]
            )
            ddq_dv[:, :root_tdim] = cs.jacobian(dv, v_symbol.sx)[:, :root_tdim] / dt

        derivatives = []
        for argument in derivative_arguments:
            fixed_torque = cs.SX.zeros(self.nv, argument.tdim)
            if argument.uid == q_symbol.uid:
                fixed_torque = ddq_dq
            elif argument.uid == v_symbol.uid:
                fixed_torque = ddq_dv
            torque_jacobian = tangent_jacobian(torque, argument)
            derivatives.append(
                (argument, dt * (fixed_torque + ddq_dtau @ torque_jacobian))
            )
        return dv, derivatives

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
        with_impulses=True,
        lifted_impulses=False,
    ):
        self.robot = robot
        self.q = q
        self.v = v
        self.frame_names = tuple(frame_names)
        self.frame_ids = tuple(robot.fmodel.getFrameId(name) for name in frame_names)

        robot.update_kinematics(q.sx, v.sx)
        self.jacobians = robot.frame_translational_jacobians(self.frame_ids)
        self.impulses = (
            [
                (moto.sym.lifted if lifted_impulses else moto.sym.inputs)(
                    f"{force_prefix}_{name}",
                    3,
                    default_val=np.asarray(force_default),
                )
                for name in self.frame_names
            ]
            if with_impulses
            else []
        )
        self.generalized_impulses = [
            jacobian.T @ force for jacobian, force in zip(self.jacobians, self.impulses)
        ]
        self.generalized_impulse = sum(self.generalized_impulses, cs.SX.zeros(robot.nv))
        self.velocities, self.heights = robot.frame_linear_kinematics(
            self.frame_ids, q.sx, v.sx
        )

        self.kinematic_gain = moto.sym.params("k_f", default_val=kinematic_gain)
        self.friction_coefficient = (
            moto.sym.params("mu", default_val=friction_coefficient)
            if self.impulses
            else None
        )
        self.kinematic_constraints = self.make_kinematic_constraints()
        self.friction_constraints = [
            self._make_friction_constraint(index) for index in range(len(self.impulses))
        ]

    def make_kinematic_constraints(self, *, name_prefix="kin"):
        return [
            self._make_kinematic_constraint(index, name_prefix)
            for index in range(len(self.frame_names))
        ]

    def _make_kinematic_constraint(self, index, name_prefix):
        velocity = self.velocities[:, index]
        residual = cs.vcat(
            [
                velocity[:2],
                self.kinematic_gain * self.heights[index] + velocity[2],
            ]
        )
        constraint = moto.constr.create(
            f"{name_prefix}_{self.frame_names[index]}",
            residual,
        )
        if self.impulses:
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
        """Add path contact constraints to an interval stage."""
        if self.friction_constraints:
            stage.add(self.friction_constraints)
        stage.add(self.kinematic_constraints)

    def add_to_endpoint(self, endpoint, constraints=None):
        endpoint.add(self.kinematic_constraints if constraints is None else constraints)

class ContactRobotModel(PinocchioCasadiModel):
    """Ready-to-use contact or acceleration-controlled robot model."""

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
        acceleration_control=False,
        lifted_contact=False,
        lifted_acceleration=False,
    ):
        super().__init__(model, name)
        self.use_fwd_dyn = use_forward_dynamics
        self.acceleration_control = acceleration_control
        self.lifted_contact = lifted_contact
        self.lifted_acceleration = lifted_contact and lifted_acceleration
        self.create_trajectory_variables(
            dt,
            q_nom,
            acceleration_control=acceleration_control,
            lifted_contact=lifted_contact,
            lifted_acceleration=self.lifted_acceleration,
        )
        self.contacts = ContactModel(
            self,
            self.q,
            self.v,
            contact_frames,
            with_impulses=not acceleration_control,
            lifted_impulses=lifted_contact,
        )

        if acceleration_control:
            self.dyn = semi_implicit_dynamics(
                f"{self.name}_acceleration",
                self.q,
                self.v,
                self.qn,
                self.vn,
                self.vn - self.v - self.a * dt,
                dt,
            )
        elif lifted_contact:
            # Built below at the implicit next state together with the contact
            # position Jacobian.
            self.rnea = None
        elif use_forward_dynamics:
            derivative_arguments = [
                self.q,
                self.v,
                self.tq,
                *self.contacts.impulses,
            ]
            self.aba, self.aba_jacobians = self.forward_dynamics_step_analytic(
                self.q_stack,
                self.v_stack,
                self.tq,
                dt,
                self.contacts.generalized_impulse,
                q_symbol=self.q,
                v_symbol=self.v,
                derivative_arguments=derivative_arguments,
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
        if lifted_contact:
            if self.lifted_acceleration:
                q_next = self.q.symbolic_integrate(self.q.sx, self.vn.sx * dt)
                self.euler_position_residual = self.q.symbolic_difference(
                    self.qn.sx, q_next
                )
                self.euler_velocity_residual = self.vn.sx - self.v.sx - self.a.sx * dt
                self.euler_residual = cs.vcat(
                    [self.euler_position_residual, self.euler_velocity_residual]
                )
                self.dyn = moto.semi_implicit_euler.create(
                    f"{self.name}_lifted_euler", self.euler_residual
                )
            self.contact_activation = [
                moto.sym.params(f"contact_active_{name}", default_val=1.0)
                for name in self.contacts.frame_names
            ]
            q_reference = (
                np.asarray(q_nom, dtype=float)
                if q_nom is not None
                else pin.neutral(self.fmodel)
            )
            reference_data = self.fmodel.createData()
            pin.forwardKinematics(self.fmodel, reference_data, q_reference)
            pin.updateFramePlacements(self.fmodel, reference_data)
            self.contact_reference = [
                moto.sym.params(
                    f"contact_reference_{name}",
                    3,
                    default_val=np.asarray(
                        reference_data.oMf[frame_id].translation, dtype=float
                    ),
                )
                for name, frame_id in zip(
                    self.contacts.frame_names, self.contacts.frame_ids
                )
            ]
            contact_rows = []
            contact_data = self.createData()
            cpin.forwardKinematics(self, contact_data, self.qn_stack, self.vn_stack)
            cpin.computeJointJacobians(self, contact_data, self.qn_stack)
            cpin.updateFramePlacements(self, contact_data)
            contact_jacobians = []
            for active, frame_id, impulse, reference in zip(
                self.contact_activation,
                self.contacts.frame_ids,
                self.contacts.impulses,
                self.contact_reference,
            ):
                jacobian = cpin.getFrameJacobian(
                    self,
                    contact_data,
                    frame_id,
                    pin.LOCAL_WORLD_ALIGNED,
                )[:3, :]
                contact_jacobians.append(jacobian)
                phi = contact_data.oMf[frame_id].translation - reference.sx
                contact = phi + (jacobian @ self.vn_stack) * self.dt
                contact_rows.append(active * contact + (1.0 - active) * impulse)
            self.contact_rows = contact_rows
            generalized_impulse = sum(
                (
                    jacobian.T @ impulse
                    for jacobian, impulse in zip(
                        contact_jacobians, self.contacts.impulses
                    )
                ),
                cs.SX.zeros(self.nv),
            )
            self.rnea = self.inverse_dynamics_impulse(
                self.qn_stack,
                self.vn_stack,
                self.a_stack,
                dt,
                generalized_impulse,
            )
            if self.lifted_acceleration:
                self.rnea_residual = (
                    self.rnea - self.generalized_torque(self.tq) * self.dt
                )
                self.lifting = moto.lifted.create(
                    f"{self.name}_rnea_contact_lifting",
                    cs.vcat(
                        [
                            self.rnea_residual,
                            *contact_rows,
                        ]
                    ),
                    [self.a, *self.contacts.impulses],
                    order=moto.approx_order.approx_order_first,
                )
            else:
                q_next = self.q.symbolic_integrate(self.q.sx, self.vn.sx * self.dt)
                self.euler_residual = self.q.symbolic_difference(self.qn.sx, q_next)
                self.rnea_residual = (
                    self.rnea - self.generalized_torque(self.tq) * self.dt
                )
                dynamics_residual = cs.vcat([self.euler_residual, self.rnea_residual])
                self.dyn = moto.semi_implicit_euler.create(
                    f"{self.name}_lifted_rnea", dynamics_residual
                )
                self.lifting = moto.lifted.create(
                    f"{self.name}_contact_lifting",
                    cs.vcat(contact_rows),
                    self.contacts.impulses,
                    order=moto.approx_order.approx_order_first,
                )
        elif not acceleration_control:
            self.dyn = self._make_contact_dynamics()
            self.lifting = None
        else:
            self.lifting = None

        self.q_nom = moto.sym.params(
            "q_nom",
            self.nq,
            default_val=q_nom if q_nom is not None else np.zeros(self.nq),
        )

    def _make_contact_dynamics(self):
        if self.use_fwd_dyn:
            dv = cs.SX.sym(f"{self.name}_analytic_dv", self.nv)
            velocity_residual = self.vn.sx - (self.v.sx + dv)
            name = f"{self.name}_fd"
        else:
            velocity_residual = self.rnea - self.generalized_torque(self.tq) * self.dt
            name = f"{self.name}_id"
        if self.configuration_velocity == "next":
            q_next = self.q.symbolic_integrate(self.q.sx, self.vn.sx * self.dt)
        else:
            name += "_predicted"
            integration_velocity = self.v.sx + dv if self.use_fwd_dyn else self.vn.sx
            q_next = self.q.symbolic_integrate(
                self.q.sx, integration_velocity * self.dt
            )
        residual_template = cs.vcat(
            [self.q.symbolic_difference(self.qn.sx, q_next), velocity_residual]
        )
        if not self.use_fwd_dyn:
            return (
                moto.semi_implicit_euler.create(name, residual_template)
                if self.configuration_velocity == "next"
                else moto.dense_dynamics.create(name, residual_template)
            )

        residual = cs.substitute(residual_template, dv, self.aba)
        dynamics = (
            moto.semi_implicit_euler.create(name, residual)
            if self.configuration_velocity == "next"
            else moto.dense_dynamics.create(name, residual)
        )
        dv_jacobian = cs.jacobian(residual_template, dv)
        aba_jacobians = {
            argument.uid: jacobian for argument, jacobian in self.aba_jacobians
        }
        derivative_arguments = [
            self.q,
            self.v,
            self.qn,
            self.vn,
            self.tq,
            *self.contacts.impulses,
        ]
        for argument in derivative_arguments:
            jacobian = tangent_jacobian(residual_template, argument)
            if argument.uid in aba_jacobians:
                jacobian += dv_jacobian @ aba_jacobians[argument.uid]
            jacobian = cs.substitute(jacobian, dv, self.aba)
            dynamics.set_analytic_jacobian(argument, jacobian)
        return dynamics

    def input_cost(
        self,
        *,
        acceleration_weight=1e-3,
        torque_weight=1e-6,
        contact_weight=1e-3,
        name=None,
    ):
        mode = (
            "acceleration"
            if self.acceleration_control
            else ("lifted_acceleration" if self.lifted_acceleration else "contact")
        )
        name = name or f"{self.name}_{mode}_input_cost"
        if self.acceleration_control:
            return moto.cost.from_vector(name, self.a, weight=2 * acceleration_weight)
        impulses = cs.vcat(self.contacts.impulses)
        residuals = [self.tq, impulses]
        weights = [
            np.full(self.tq.numel(), 2 * torque_weight),
            np.full(impulses.numel(), 2 * contact_weight),
        ]
        if self.lifted_acceleration:
            residuals.append(self.a)
            weights.append(np.full(self.a.numel(), 2 * acceleration_weight))
        return moto.cost.from_vector(
            name, cs.vcat(residuals), weight=np.concatenate(weights)
        )


class ViserRobot:
    """Small Viser URDF wrapper for fixed- and floating-base trajectories."""

    def __init__(self, urdf, *, floating_base=False, root="/robot", port=8080):
        import viser
        from viser.extras import ViserUrdf

        self.server = viser.ViserServer(port=port)
        self.server.scene.set_up_direction("+z")
        self.server.scene.add_grid("/ground", infinite_grid=True)
        self.root = self.server.scene.add_frame(root, show_axes=False)
        self.robot = ViserUrdf(self.server, Path(urdf), root_node_name=root)
        self.floating_base = floating_base
        self.joint_count = len(self.robot.get_actuated_joint_names())

    def update(self, configuration):
        q = np.asarray(configuration)
        with self.server.atomic():
            if self.floating_base:
                self.root.position = q[:3]
                self.root.wxyz = q[[6, 3, 4, 5]]
            self.robot.update_cfg(q[-self.joint_count :])

    def add_target(self, name, position, *, xyzw=None, color=(0, 255, 0)):
        position = np.asarray(position)
        path = f"/targets/{name}"
        self.server.scene.add_icosphere(
            f"{path}/point", radius=0.025, color=color, position=position
        )
        return self.server.scene.add_frame(
            f"{path}/frame",
            position=position,
            wxyz=(1.0, 0.0, 0.0, 0.0)
            if xyzw is None
            else np.asarray(xyzw)[[3, 0, 1, 2]],
            axes_length=0.12,
            axes_radius=0.006,
        )


def animate_trajectory(viewer, configurations, time_steps):
    """Replay a trajectory continuously on a Viser server."""
    import time

    while True:
        for index, configuration in enumerate(configurations):
            start = time.perf_counter()
            viewer.update(configuration)
            if index < len(time_steps):
                remaining = time_steps[index] - (time.perf_counter() - start)
                if remaining > 0:
                    time.sleep(remaining)
        time.sleep(0.5)


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
