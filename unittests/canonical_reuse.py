import moto


calls = {"precompute": 0, "function": 0}
x0, _ = moto.sym.states("canonical_reuse_x0", 1)
x1, _ = moto.sym.states("canonical_reuse_x1", 1)


def first_outputs():
    calls["precompute"] += 1
    return (x0.sx * x0.sx,)


cache0 = moto.precompute.canonical(
    "canonical_reuse_square", ("square", 1), (x0,), first_outputs
)
cache1 = moto.precompute.canonical(
    "canonical_reuse_square",
    ("square", 1),
    (x1,),
    lambda: (_ for _ in ()).throw(RuntimeError("factory was not lazy")),
)


def first_function(name):
    calls["function"] += 1
    return moto.constr.create(name, cache0[0])


function0 = moto.func.canonical(
    "canonical_reuse_consumer", ("identity", 1), cache0, first_function
)
function1 = moto.func.canonical(
    "canonical_reuse_consumer",
    ("identity", 1),
    cache1,
    lambda _: (_ for _ in ()).throw(RuntimeError("factory was not lazy")),
)

assert calls == {"precompute": 1, "function": 1}
assert cache0[0].uid != cache1[0].uid
assert function0.name == function1.name
assert x0.uid in {argument.uid for argument in function0.in_args}
assert x1.uid in {argument.uid for argument in function1.in_args}
