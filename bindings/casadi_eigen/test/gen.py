import pinocchio as pin
import pinocchio.casadi as cpin
import casadi as cs

# Load the robot model
model_path = "/home/harper/Documents/atri/rsc/go2_description/urdf"
urdf_filename = "go2_description.urdf"
urdf_path = f"{model_path}/{urdf_filename}"

# Build the Pinocchio model
model = pin.buildModelFromUrdf(urdf_path)
data = model.createData()

# Convert to CasADi Pinocchio model
cpin_model = cpin.Model(model)
cpin_data = cpin_model.createData()

q = cs.SX.sym("q", model.nq)
v = cs.SX.sym("v", model.nv)
a = cs.SX.sym("a", model.nv)

tau = cpin.rnea(cpin_model, cpin_data, q, v, a)

from atri.codegen import *

generate_cpp_with_eigen_ref_vector(
    "rnea",
    [q, v, a],
    [tau],
    "gen",
)

# # Generate CasADi function
# rnea_function = cs.Function("f", [q, v, a], [tau])

# # Generate C code
# rnea_function.generate("f.cpp", {"cpp": False})