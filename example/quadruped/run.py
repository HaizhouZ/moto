import moto
# q = moto.sym("q", 12, field=moto.field_x)
# moto.sym_impl()
# q2 = moto.func("q2", q, 2 * q, field=moto.field_eq_x)

q, q_next = moto.sym.states("q", 12)
print("------------------------------")
# print(q.sym_base.as_expr().get_impl())
# b = q.sym_base

# print(b.as_expr().get_impl())

# print(moto.get_sym_sx(q.sym_base))
# print(moto.get_sym_sx(b))

moto.get_all_sx([q, q_next])
c = moto.func("c", [q, q_next], 2 * q)
print(q.use_count, q.impl_use_count)
print(q.sym_base)
print(q.next)
print(q_next.prev)

q_next.name = "q_next111"
print(q.next.sym_base.as_expr().name)