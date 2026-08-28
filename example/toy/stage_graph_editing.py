#!/usr/bin/env python3

import moto


x, xn = moto.sym.states("stage_edit_x", 1)
u = moto.sym.inputs("stage_edit_u", 1)

dynamics = moto.dense_dynamics.create(
    "stage_edit_dynamics",
    xn.sx - x.sx - u.sx,
)
base_cost = moto.cost.from_scalar("stage_edit_base_cost", u)
changed_cost = moto.cost.from_scalar("stage_edit_changed_cost", x)
replacement_cost = moto.cost.from_scalar("stage_edit_replacement_cost", u, weight=2.0)
tail_cost = moto.cost.from_scalar("stage_edit_tail_cost", u, weight=3.0)


def make_stage(cost):
    stage = moto.stage()
    stage.add([dynamics, cost])
    return stage


def main():
    sqp = moto.sqp(n_job=1)
    stage = make_stage(base_cost)
    sqp.stages.extend([stage.copy() for _ in range(5)])
    stages = sqp.stages
    graph_end = sqp.ed

    # 1. Edit a graph-owned stage directly through its retained pointer.
    stages[1].add(changed_cost)
    assert len(sqp.nodes) == 5

    # 2. Use ordinary list slicing to replace a stage range.
    replacement = make_stage(replacement_cost)
    keep_first, keep_last = stages[0], stages[-1]
    del stages[1:3]
    stages.insert(1, replacement.copy())
    assert sqp.stages[0] is keep_first
    assert sqp.stages[-1] is keep_last

    # 3. Shift retained pointers and append only a newly copied tail stage.
    retained = list(sqp.stages[1:])
    del sqp.stages[0]
    sqp.stages.append(make_stage(tail_cost).copy())
    assert all(actual is expected for actual, expected in zip(sqp.stages[:-1], retained))
    assert sqp.ed.stage is graph_end.stage
    assert len(sqp.nodes) == 4

    print("native stage-list editing passed")


if __name__ == "__main__":
    main()
