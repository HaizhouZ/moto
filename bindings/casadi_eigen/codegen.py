import casadi as ca
import os
import re


def generate_cpp_with_eigen_ref_vector(func_name, sx_inputs, sx_outputs, output_dir="gen"):
    # Step 1: Create CasADi function
    casadi_func = ca.Function(func_name, sx_inputs, sx_outputs)

    # Step 2: Generate raw C code
    cgen = ca.CodeGenerator("raw")
    cgen.add(casadi_func)
    os.makedirs(output_dir, exist_ok=True)
    cpp_path = os.path.join(output_dir, f"{func_name}_")
    cgen.generate(cpp_path)

    # Step 3: Parse raw C code and replace array access with Eigen::Ref access
    with open(cpp_path + "raw.c", "r") as f:
        lines = f.readlines()

    # Extract input/output variable shapes
    input_shapes = [x.shape for x in sx_inputs]
    output_shapes = [y.shape for y in sx_outputs]

    # Create mapping from flat index to Eigen::Ref(i,j)
    def flat_index_to_ij(index, shape):
        rows, cols = shape
        i = index % rows
        j = index // rows
        return i, j

    def simplify_conditional(code_str):
        pattern = r"arg\[(\d+)\]\? ([^:;]+) : 0;"
        return re.sub(pattern, r"\2;", code_str)

    def simplify_if(code_str):
        pattern = r"if\s*\(res\[0\]\s*!=\s*0\)\s*\s*(.+);"
        return re.sub(pattern, r"\1;", code_str)

    def make_input_ref_access(arg_idx, index):
        i, j = flat_index_to_ij(index, input_shapes[arg_idx])
        return f"inputs[{arg_idx}]({i},{j})"

    def make_output_ref_access(arg_idx, index):
        i, j = flat_index_to_ij(index, output_shapes[arg_idx])
        return f"outputs[{arg_idx}]({i},{j})"

    # Step 4: Replace all casadi generated arrays with Eigen::Ref notation
    processed_lines = []
    func_found = False
    func_done = False
    for line in lines:
        if not func_found:
            if "static int casadi_f0" in line:
                processed_lines.append(f"CASADI_SYMBOL_EXPORT void {func_name}(\n")
                processed_lines.append("    std::vector<Eigen::Ref<Eigen::VectorXd>>& inputs,\n")
                processed_lines.append("    std::vector<Eigen::Ref<Eigen::MatrixXd>>& outputs) {\n")
                func_found = True
            continue

        if "return 0;" in line:
            continue
        if "}" in line and not func_done:
            processed_lines.append("}\n")
            func_done = True
            break

        line = simplify_conditional(line)
        line = simplify_if(line)

        # Replace input array access: arg[i][index] -> inputs[i](r,c)
        line = re.sub(r"arg\[(\d+)\]\[(\d+)\]", lambda m: make_input_ref_access(int(m[1]), int(m[2])), line)

        # Replace output array access: res[i][index] -> outputs[i](r,c)
        line = re.sub(r"res\[(\d+)\]\[(\d+)\]", lambda m: make_output_ref_access(int(m[1]), int(m[2])), line)

        processed_lines.append(line)

    # Step 5: Write new C++ file with Eigen interface
    final_cpp_path = os.path.join(output_dir, f"{func_name}.cpp")
    with open(final_cpp_path, "w") as f:
        f.write("#include <vector>\n#include <Eigen/Dense>\n\n")
        f.write("#define casadi_real double\n\n")
        f.write('#ifdef __cplusplus\n')
        # f.write("#ifndef CASADI_SYMBOL_EXPORT\n")
        f.write("#if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)\n")
        f.write("    #if defined(STATIC_LINKED)\n")
        f.write("    #define CASADI_SYMBOL_EXPORT\n")
        f.write("    #else\n")
        f.write("    #define CASADI_SYMBOL_EXPORT __declspec(dllexport)\n")
        f.write("    #endif\n")
        f.write("#elif defined(__GNUC__)\n")
        f.write("    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility (\"default\")))\n")
        f.write("#endif\n")
        # f.write("#endif\n")
        f.write('extern "C" {\n')
        f.write('#endif\n\n')
        f.writelines(processed_lines)
        f.write('\n\n#ifdef __cplusplus\n')
        f.write("}\n")
        f.write('#endif\n\n')

    print(f"C++ code generated: {final_cpp_path}")
    # Step 6: Compile the generated C++ code into a shared library
    so_file_path = os.path.join(output_dir, f"lib{func_name}.so")
    compile_command = f"g++ -shared -fPIC -O3 -o  {so_file_path} {final_cpp_path} -I /usr/include/eigen3"
    os.system(compile_command)
    print(f"Shared library generated: {so_file_path}")
