import moto
import copy
# q = moto.sym("q", 12, field=moto.field_x)
# moto.sym_impl()
# q2 = moto.func("q2", q, 2 * q, field=moto.field_eq_x)

q, q_next = moto.sym.states("q", 12)
v = moto.sym.inputs("v", 12)
print("------------------------------")
# print(q.sym_base.as_expr().get_impl())
# b = q.sym_base

# print(b.as_expr().get_impl())

# print(moto.get_sym_sx(q.sym_base))
# print(moto.get_sym_sx(b))

moto.print_all_sx([q, q_next])
prob = moto.ocp.create()
prob.add([q, q_next])
d = moto.constr("d", [q, q_next], q.T @ q_next, field=moto.field_eq_x).as_eq()
# print(c.uid)
# e = moto.expr()
# c.as_eq()#.as_ineq().as_constr()
# print(q.use_count, q_next.use_count)
# print(q.sym_base)
# print(q.next)
# print(q_next.prev)

# q_next.name = "q_next111"
# q_next.dim = 13

print(d.uid)
print(prob.exprs(moto.field.field_x))

print("------------------------------")
exit(0)
# prob = moto.ocp.create()
# prob.add([q, q_next, c])
# print(c.use_count)
# print(prob.exprs(moto.field.field_x))
# p = moto.func.cost("cost_test", [q, v], q.T @ v)
# pt = p.as_terminal()
# print(p.uid)
# print(pt.uid)
# print(p.name)

# a = prob.exprs(moto.field_eq_xu)
# c.name = "f_test_updated"
# print(a[0].as_expr().name)