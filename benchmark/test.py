import moto
import numpy as np

u = moto.sym.inputs("u", 3, np.array([0.1, 0.2, 0.3]))
b = moto.sym.params("b", 3, np.array([1.0, 1.0, 1.0]))
prob = moto.ocp.create()
prob2 = moto.ocp.create()
prob.add(u)
prob2.add(u)
prob.add([u, b])
prob2.add([u, b])
print([u.__sym__])
print(prob.exprs(moto.field_u))
# print(prob.exprs(moto.field_p))
# print(prob2.exprs(moto.field_u))
print("---")
u2 = moto.var(prob.exprs(moto.field_u)[0])
del prob
del prob2

print([u.__sym__])
print(u2.__sym__)
print("-----")
f = moto.constr.create("f", [u, b], u + b,moto.approx_order_first, moto.field_u)
z = moto.sym.params("z", 2, np.array([5.0, 10.0]))
f.add_argument(z)
print(f.uid)
print(f.in_args)