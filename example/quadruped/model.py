import casadi as cs
import moto
import numpy as np

from example.helpers import (
    ContactRobotModel,
    GO2_FOOT_FRAMES,
)


GAIT_ORDER = (0, 3, 1, 2)
GAIT_SETTING = (1, 1, 0, 0)


class QuadrupedModel(ContactRobotModel):
    def __init__(self, model, *, foot_frames=GO2_FOOT_FRAMES, **kwargs):
        super().__init__(model, contact_frames=foot_frames, **kwargs)

    def prepare_problem_terms(self, *, state_cost_name="c"):
        self.joint_limit_constr = self.joint_limit_constraint(self.q, self.v)
        self.terminal_joint_limit_constr = self.joint_limit_constraint(
            self.q, self.v, name="q_limit_terminal"
        )
        self.input_limit_constr = (
            moto.ineq.bounds("a_limit", self.a, -50.0, 50.0)
            if self.acceleration_control
            else self.torque_limit_constraint(self.tq)
        )
        self.state_cost = self._state_cost(state_cost_name)
        self.terminal_state_cost = self._state_cost(f"{state_cost_name}_terminal")
        self.terminal_contact_constraints = (
            []
            if self.lifted_contact
            else self.contacts.make_kinematic_constraints(name_prefix="kin_terminal")
        )
        self.swing_force_constraints = [
            moto.constr.create(f"swing_force_{name}", force.sx)
            for name, force in zip(self.contacts.frame_names, self.contacts.impulses)
        ]

    def _state_cost(self, name):
        q_nom_res = self.q.symbolic_difference(self.q.sx, self.q_nom.sx)
        residual = cs.vcat([q_nom_res, self.v_stack])
        weight = np.r_[
            np.full(6, 200.0),
            np.full(q_nom_res.numel() - 6, 2.0),
            np.full(6, 2.0),
            np.full(self.v_stack.numel() - 6, 0.02),
        ]
        return moto.cost.from_vector(name, residual, weight=weight)

    def create_stage(self, nominal_dt):
        stage = moto.stage()
        stage.add([self.dyn, self.input_limit_constr, self.input_cost()])
        if self.lifted_contact:
            stage.add(self.contacts.friction_constraints)
        else:
            self.contacts.add_to_stage(stage)
        if isinstance(self.dt, cs.SX):
            bounds = moto.sym.params(
                "dt_bound", 2, default_val=np.array([1e-4, 5e-2])
            )
            stage.add(
                [
                    moto.ineq.create("dt", self.dt.sx, bounds[0], bounds[1]),
                    moto.cost.from_scalar(
                        "c_t", self.dt - nominal_dt, weight=2e8
                    ),
                ]
            )
        stage.add([self.joint_limit_constr, self.state_cost])
        return stage

    def add_terminal_terms(self, endpoint):
        if not self.lifted_contact:
            self.contacts.add_to_endpoint(endpoint, self.terminal_contact_constraints)
        endpoint.add(
            [self.terminal_joint_limit_constr, self.terminal_state_cost]
        )


def swing_feet(step):
    return tuple(
        foot
        for index, foot in enumerate(GAIT_ORDER)
        if (not GAIT_SETTING[index] if step % 2 == 0 else bool(GAIT_SETTING[index]))
    )


def _phase_stage(stage, model, step):
    disabled = []
    feet = swing_feet(step)
    if model.lifted_contact:
        disabled.extend(model.contacts.friction_constraints[foot] for foot in feet)
    elif model.acceleration_control:
        disabled.extend(model.contacts.kinematic_constraints[foot] for foot in feet)
    else:
        for foot in feet:
            disabled.append(model.contacts.kinematic_constraints[foot])
            disabled.append(model.contacts.friction_constraints[foot])
    phase = stage.with_status(moto.active_status_config(deactivate_list=disabled))
    if not model.acceleration_control and not model.lifted_contact:
        phase.add([model.swing_force_constraints[foot] for foot in feet])
    return phase


def add_gait_phases(sqp, stage, model, *, horizon, steps, nodes_per_step):
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if steps < 0 or nodes_per_step < 1:
        raise ValueError(
            "steps must be nonnegative and nodes_per_step must be positive"
        )

    gait_stages = steps * nodes_per_step
    if gait_stages > horizon:
        raise ValueError("horizon must be at least steps * nodes-per-step")

    remaining = horizon - gait_stages
    head_stance = remaining // 2
    tail_stance = remaining - head_stance
    phases = []
    if head_stance:
        phases.append((stage, head_stance))
    phases.extend(
        (_phase_stage(stage, model, step), nodes_per_step)
        for step in range(1, steps + 1)
    )
    if tail_stance:
        phases.append((stage, tail_stance))

    first = len(sqp.stages)
    for phase, count in phases:
        sqp.stages.extend([phase.copy() for _ in range(count)])
    if len(sqp.stages) - first != horizon:
        raise RuntimeError(
            "gait phase construction did not match the requested horizon"
        )
    return head_stance, tail_stance
