import gdb

def run():
    gdb.execute("run")
    gdb.execute("frame 0")
    print("d.d_u.K size:", gdb.execute("p d.d_u.K.rows()", to_string=True).strip(), "x", gdb.execute("p d.d_u.K.cols()", to_string=True).strip())
    print("d.d_u.k size:", gdb.execute("p d.d_u.k.rows()", to_string=True).strip(), "x", gdb.execute("p d.d_u.k.cols()", to_string=True).strip())
    print("trial_prim_step[__x] size:", gdb.execute("p d.trial_prim_step[0].rows()", to_string=True).strip())
    print("trial_prim_step[__u] size:", gdb.execute("p d.trial_prim_step[2].rows()", to_string=True).strip())
    gdb.execute("bt")

run()
