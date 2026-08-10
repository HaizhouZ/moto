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
        def jac(equation, variable):
            return system.jac(equation, variable)

        forces = robot.contacts.impulses

        if robot.lifted_acceleration:
            euler_q = jac(robot.euler_residual, robot.qn).mx[:nv, :]
            euler_v = jac(robot.euler_residual, robot.vn).mx
            position_v = euler_v[:nv, :]
            velocity_v = euler_v[nv:, :]
            velocity_a = jac(robot.euler_residual, robot.a).mx[nv:, :]

            acceleration_diagonal = cs.diag(velocity_a)

            def eliminate_acceleration(rhs):
                return rhs / cs.repmat(
                    acceleration_diagonal, 1, rhs.size2()
                )

            a_v = eliminate_acceleration(velocity_v)
            rnea_a = jac(robot.rnea_residual, robot.a).mx
            rnea_q = jac(robot.rnea_residual, robot.qn).mx
            rnea_v = (
                jac(robot.rnea_residual, robot.vn).mx - rnea_a @ a_v
            )
            rnea_f = cs.horzcat(
                *[jac(robot.rnea_residual, force).mx for force in forces]
            )

            contact_q = cs.vertcat(
                *[jac(contact, robot.qn).mx for contact in robot.contact_rows]
            )
            contact_v = cs.vertcat(
                *[jac(contact, robot.vn).mx for contact in robot.contact_rows]
            )
            regularizer = None
            contact_blocks = []
            for row, contact in enumerate(robot.contact_rows):
                blocks = []
                for col, force in enumerate(forces):
                    block = jac(contact, force)
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

            x_columns = system.h_x().size2()
            u_columns = system.h_u().size2()
            projection = solve(
                cs.horzcat(system.h_x(), system.h_u(), system.h())
            )
            return moto.lifted.elimination(
                projection[:, :x_columns],
                projection[:, x_columns:x_columns + u_columns],
                projection[:, x_columns + u_columns:],
                [],
                solve(system.action_rhs),
            )

        aqq = jac(robot.euler_residual, robot.qn).mx
        aqv = jac(robot.euler_residual, robot.vn).mx
        avq = jac(robot.rnea_residual, robot.qn).mx
        avv = jac(robot.rnea_residual, robot.vn).mx
        bq = cs.horzcat(
            *[jac(robot.euler_residual, force).mx for force in forces]
        )
        bv = cs.horzcat(
            *[jac(robot.rnea_residual, force).mx for force in forces]
        )
        cq = cs.vertcat(
            *[jac(contact, robot.qn).mx for contact in robot.contact_rows]
        )
        cv = cs.vertcat(
            *[jac(contact, robot.vn).mx for contact in robot.contact_rows]
        )
        regularizer = None
        contact_blocks = []
        for row, contact in enumerate(robot.contact_rows):
            blocks = []
            for col, force in enumerate(forces):
                block = jac(contact, force)
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

        response_x = solve(system.h_x())
        response_u = solve(system.h_u())
        response_residual = solve(system.h())
        response_action = solve(system.action_rhs)
        return moto.lifted.elimination(
            response_x,
            response_u,
            response_residual,
            [],
            response_action,
        )

    robot.dyn.add_subconstraint(robot.lifting)
    robot.dyn = robot.dyn.set_elimination_graph(elimination)
    return robot.dyn
