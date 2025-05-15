import pinocchio as pin
import numpy as np

# Load the robot model
model_path = "/home/harper/Documents/atri/rsc/go2_description/urdf"
urdf_filename = "go2_description.urdf"
urdf_path = f"{model_path}/{urdf_filename}"

# Build the Pinocchio model
model = pin.buildModelFromUrdf(urdf_path)
data = model.createData()
q = np.random.random(model.nq)
v = np.random.random(model.nv)
a = np.random.random(model.nv)
tau = np.zeros((model.nv,1))

print("origin")
print(pin.rnea(model, data, q, v, a))

import sys
sys.path.append('/home/harper/Documents/atri/build/py_casadi_eigen')
import python_invoke_interface as invoke

func = invoke.load_vec_vec("gen/librnea.so", "rnea")
print("loaded")
# Call the function
print("call")
invoke.invoke_vec_vec(func, [q, v, a], [tau])
print("done")

print("compiled")
print(tau)
# # Generate CasADi function
# rnea_function = cs.Function("f", [q, v, a], [tau])

# # Generate C code
# rnea_function.generate("f.cpp", {"cpp": False})
