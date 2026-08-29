"""Structured elimination graph for implicit Euler + RNEA/contact lifting."""

import casadi as cs

import moto


def _configuration_solver(system, matrix):
    diagonal = cs.diag(matrix)
    orientation = system.solve(matrix[3:6, 3:6], spd=True)

    def solve(rhs):
        if rhs.nnz() == 0:
            return cs.MX(cs.Sparsity(matrix.size1(), rhs.size2()))

        def diagonal_solve(begin, end):
            return rhs[begin:end, :] / cs.repmat(
                diagonal[begin:end], 1, rhs.size2()
            )

        return cs.vertcat(
            diagonal_solve(0, 3),
            orientation.solve(rhs[3:6, :]),
            diagonal_solve(6, matrix.size1()),
        )

    return solve


def _semi_implicit_euler_solver(system, matrix):
    size = matrix.size1()
    configuration_dim = size // 2
    configuration = _configuration_solver(
        system, matrix[:configuration_dim, :configuration_dim]
    )

    def solve(rhs):
        q_rhs = (
            rhs[:configuration_dim, :]
            - matrix[:configuration_dim, configuration_dim:]
            @ rhs[configuration_dim:, :]
        )
        return cs.vertcat(
            configuration(q_rhs), rhs[configuration_dim:, :]
        )

    return solve


def configure_lifted_contact_elimination(robot, regularization=1e-9):
    """Attach one reusable elimination graph to a lifted contact model."""
    if robot.lifting is None:
        raise ValueError("robot has no lifted RNEA/contact constraint")
    nv = robot.nv

    def elimination(system):
        forces = robot.contacts.impulses

        def dyn_jac(variable):
            return system.jac(robot.dyn, variable).mx

        def lift_jac(variable):
            return system.jac(robot.lifting, variable).mx

        if robot.lifted_acceleration:
            euler_q = dyn_jac(robot.qn)[:nv, :]
            euler_v = dyn_jac(robot.vn)
            position_v = euler_v[:nv, :]
            velocity_v = euler_v[nv:, :]
            velocity_a = dyn_jac(robot.a)[nv:, :]

            acceleration_diagonal = cs.diag(velocity_a)

            def eliminate_acceleration(rhs):
                return rhs / cs.repmat(
                    acceleration_diagonal, 1, rhs.size2()
                )

            a_v = eliminate_acceleration(velocity_v)
            rnea_a = lift_jac(robot.a)[:nv, :]
            rnea_q = lift_jac(robot.qn)[:nv, :]
            rnea_v = lift_jac(robot.vn)[:nv, :] - rnea_a @ a_v
            rnea_f = cs.horzcat(
                *[lift_jac(force)[:nv, :] for force in forces]
            )

            contact_q = lift_jac(robot.qn)[nv:, :]
            contact_v = lift_jac(robot.vn)[nv:, :]
            regularizer = None
            contact_blocks = []
            force_blocks = [
                system.jac(robot.lifting, force) for force in forces
            ]
            for row in range(len(forces)):
                blocks = []
                for col, force_block in enumerate(force_blocks):
                    block = force_block.rows(
                        nv + 3 * row, nv + 3 * (row + 1)
                    )
                    if row == col:
                        if regularizer is None:
                            regularizer = block.param(
                                regularization, "contact_regularization"
                            )
                        blocks.append(block.add_diag(regularizer))
                    else:
                        blocks.append(block.mx)
                contact_blocks.append(cs.horzcat(*blocks))
            contact_f = cs.vertcat(*contact_blocks)

            configuration = _configuration_solver(system, euler_q)
            q_v = configuration(position_v)
            rnea_q_solve = rnea_q
            mass = rnea_v - rnea_q_solve @ q_v
            force = rnea_f
            contact_q_solve = contact_q
            contact_v_schur = contact_v - contact_q_solve @ q_v
            mass_factor = system.solve(mass)
            mass_force = mass_factor.solve(force)
            contact_factor = system.solve(
                contact_f - contact_v_schur @ mass_force
            )

            def solve(rhs):
                acceleration_base = eliminate_acceleration(rhs[nv:2 * nv, :])
                q_rhs = rhs[:nv, :]
                rnea_rhs = rhs[2 * nv:3 * nv, :] - rnea_a @ acceleration_base
                contact_rhs = rhs[3 * nv:, :]
                q_base = configuration(q_rhs)
                velocity_base = mass_factor.solve(
                    rnea_rhs - rnea_q_solve @ q_base
                )
                force_step = contact_factor.solve(
                    contact_rhs
                    - contact_q_solve @ q_base
                    - contact_v_schur @ velocity_base
                )
                velocity = velocity_base - mass_force @ force_step
                configuration_step = q_base - q_v @ velocity
                acceleration = acceleration_base - a_v @ velocity
                return cs.vertcat(
                    configuration_step, velocity, acceleration, force_step
                )

            return system.eliminate(solve)

        aqq = dyn_jac(robot.qn)[:nv, :]
        aqv = dyn_jac(robot.vn)[:nv, :]
        avq = dyn_jac(robot.qn)[nv:, :]
        avv = dyn_jac(robot.vn)[nv:, :]
        bq = cs.horzcat(
            *[dyn_jac(force)[:nv, :] for force in forces]
        )
        bv = cs.horzcat(
            *[dyn_jac(force)[nv:, :] for force in forces]
        )
        cq = lift_jac(robot.qn)
        cv = lift_jac(robot.vn)
        regularizer = None
        contact_blocks = []
        force_blocks = [system.jac(robot.lifting, force) for force in forces]
        for row in range(len(forces)):
            blocks = []
            for col, force_block in enumerate(force_blocks):
                block = force_block.rows(3 * row, 3 * (row + 1))
                if row == col:
                    if regularizer is None:
                        regularizer = block.param(
                            regularization, "contact_regularization"
                        )
                    blocks.append(block.add_diag(regularizer))
                else:
                    blocks.append(block.mx)
            contact_blocks.append(cs.horzcat(*blocks))
        d = cs.vertcat(*contact_blocks)

        configuration_solve = _configuration_solver(system, aqq)
        qv = configuration_solve(aqv)
        qf = configuration_solve(bq)
        mass = avv - avq @ qv
        force = bv - avq @ qf
        contact_v = cv - cq @ qv
        contact_f = d - cq @ qf
        mass_factor = system.solve(mass)
        mass_force = mass_factor.solve(force)
        contact_schur = contact_f - contact_v @ mass_force
        contact_factor = system.solve(contact_schur)

        def solve(rhs):
            q_base = configuration_solve(rhs[:nv, :])
            velocity_rhs = rhs[nv : 2 * nv, :] - avq @ q_base
            contact_rhs = rhs[2 * nv :, :] - cq @ q_base
            velocity_base = mass_factor.solve(velocity_rhs)
            force_step = contact_factor.solve(
                contact_rhs - contact_v @ velocity_base
            )
            velocity = velocity_base - mass_force @ force_step
            configuration = q_base - qv @ velocity - qf @ force_step
            return cs.vertcat(configuration, velocity, force_step)

        return system.eliminate(solve)

    robot.dyn = robot.dyn.with_elimination_graph(
        elimination, [robot.lifting]
    )
    return robot.dyn
