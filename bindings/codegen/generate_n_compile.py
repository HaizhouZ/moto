import casadi as cs
import os
import re
import numpy as np
from multiprocessing import Process

process_ = []


def filter_func_near_zero(func_name: str, inputs: list[cs.SX], expr: list[cs.SX], tol=1e-8, ntrials=50):
    """
    The nonzeros across all trials will be recorded to generate a new function,
    of which the output is filtered by zeros in the random tests
    """
    f = cs.Function(func_name, inputs, expr)
    nz_cnt = [np.zeros((e.shape[0], e.shape[1]), dtype=np.int64) for e in expr]
    for n in range(ntrials):
        inputs_data = [2 * (cs.DM.rand(s.shape) - 0.5) for s in inputs]
        if isinstance(expr, list):
            res = f(*inputs_data)
            if not isinstance(res, tuple):
                res = (res,)
            for k in range(len(expr)):
                for i in range(expr[k].shape[0]):
                    for j in range(expr[k].shape[1]):
                        # mark if higher than tol
                        if abs(res[k][i, j]) > tol:
                            nz_cnt[k][i, j] += 1
    filtered = [cs.SX.zeros(cs.Sparsity(e.shape[0], e.shape[1])) for e in expr]
    # for each output, find the true nonzeros that passed all trials
    for k in range(len(expr)):
        non_zero_pairs = np.argwhere(nz_cnt[k] // ntrials == 1)
        for i, j in non_zero_pairs:
            filtered[k][i, j] = expr[k][i, j]

    return cs.Function(func_name, inputs, filtered)


def ccs_index_to_ij(rows, cols, row, colind):
    """
    create index-ij pair map from ccs format sparsity (default of casadi)
    """
    ij_pairs = []
    for j in range(cols):  # Loop over columns
        for k in range(colind[j], colind[j + 1]):
            i = row[k]
            ij_pairs.append((i, j))
    return ij_pairs

import pprint

def make_func_json(func_name, inputs: list[cs.SX], outputs: list[cs.SX], output_dir="gen"):
    """
    create a json file for the function
    """
    import json

    os.makedirs(output_dir, exist_ok=True)
    func_json = {
        "name": func_name,
        "inputs": {e.name: e.shape[0] for e in inputs},
        "outputs": [e.shape for e in outputs],
    }

    with open(os.path.join(output_dir, f"{func_name}.json"), "w") as f:
        # Use json to write the function metadata
        json.dump(func_json, f, indent=4)


def generate_and_compile(
    func_name,
    sx_inputs: list[cs.SX],
    sx_output: cs.SX,
    output_dir="gen",
    keep_raw=False,
    keep_c_src=False,
    compile=True,
    exclude=[],
    gen_eval=True,
    gen_jacobian=False,
    gen_hessian=False,
    append_value=False,
    append_jac=False,
    ext_jac: list[tuple[cs.SX, cs.SX]] = [],  # [(input, jacobian)]
    ext_hess: list[tuple[cs.SX, cs.SX, cs.SX]] = [],  # [input1, input2, jacobian)]
    check_jac_ad: bool = False,
    check_hess_ad: bool = False,
):
    """
    generate and compile the evaluation and derivatives of a function
    will create async threads of compilation
    call wait_until_generated() to ensure the compilation is done
    """
    worker = []
    if gen_eval:
        worker = [(func_name, sx_inputs, [sx_output], dict(append=append_value))]
    excluded = [e.name for e in exclude]
    external_jac = {in_arg.name: jac for (in_arg, jac) in ext_jac}
    external_hess = {(arg0.name, arg1.name): hess for (arg0, arg1, hess) in ext_hess}
    if ext_jac:
        gen_jacobian = True
    if ext_hess:
        gen_hessian = True
    vjp_for_hess = False

    if sx_output.shape[0] > 1 and not ext_hess and gen_hessian:
        # will use jacobian to compute vjp -> hessian
        vjp_for_hess = True

    jacs = []
    # generate jacobian
    if gen_jacobian or vjp_for_hess:
        for s in sx_inputs:
            if s.name not in excluded:
                if external_jac and s.name in external_jac.keys():
                    jacs.append(external_jac[s.name])
                    continue
                jacs.append(cs.jacobian(sx_output, s))
            else:
                jacs.append(cs.SX(0))
        if gen_jacobian and jacs:
            if check_jac_ad:
                f_ad = cs.Function(func_name + "_ad", sx_inputs, [cs.jacobian(sx_output, e) for e in sx_inputs])
                worker.append(
                    (func_name + "_jac", sx_inputs, jacs, dict(check=True, append=append_jac, f_ground_truth=f_ad))
                )
            else:
                worker.append((func_name + "_jac", sx_inputs, jacs, dict(append=append_jac)))

    # generate hessian
    hess = []
    if gen_hessian:
        # use AD of vjp to compute hessian
        if vjp_for_hess:
            lbd = cs.SX.sym(func_name + "_lbd", sx_output.shape[0])
            for idx_i, i in enumerate(sx_inputs):
                hess.append([])
                if i.name not in excluded:
                    vjp_ = cs.jtimes(sx_output, i, lbd, True)
                for idx_j, j in enumerate(sx_inputs):
                    if i.name in excluded or j.name in excluded or i.field.value < j.field.value:
                        hess[-1].append(cs.SX(0))
                        continue
                    # for i,j in same field, just copy
                    if i.field.value == j.field.value and idx_i > idx_j:
                        hess[-1].append(hess[idx_j][idx_i].T)
                        continue
                    hess[-1].append(cs.jacobian(vjp_, j))
            hess_inputs = sx_inputs + [lbd]
        else:
            for i in sx_inputs:
                hess.append([])
                for j in sx_inputs:
                    if i.name in excluded or j.name in excluded or (i.field.value < j.field.value):
                        hess[-1].append(cs.SX(0))
                        continue
                    if external_hess:
                        if (i.name, j.name) in external_hess.keys():
                            hess[-1].append(external_hess[(i.name, j.name)])
                            continue
                        elif (j.name, i.name) in external_hess.keys():
                            hess[-1].append(external_hess[(j.name, i.name)])
                            continue
                    # if not included will do autodiff
                    hess[-1].append(cs.jacobian(cs.jacobian(sx_output, i), j))
            hess_inputs = sx_inputs
        if hess:
            hess = [item for sublist in hess for item in sublist]
            worker.append((func_name + "_hess", hess_inputs, hess, dict(append=True)))

    def impl(
        func_name,
        sx_inputs: list[cs.SX],
        sx_outputs: list[cs.SX],
        check: bool = False,
        append: bool = False,
        f_ground_truth: cs.Function | None = None,
    ):
        n_in = len(sx_inputs)
        # Step 1: Create CasADi function, filter zeros
        # casadi_func = cs.Function(func_name, sx_inputs, sx_outputs)
        casadi_func = filter_func_near_zero(func_name, sx_inputs, sx_outputs)
        sx_outputs = casadi_func(*sx_inputs)
        if not isinstance(sx_outputs, tuple):
            sx_outputs = (sx_outputs,)

        if check:
            assert isinstance(f_ground_truth, cs.Function)
            check_res_inf = np.zeros(n_in)
            # do 20 trials and see if matches the ground truth
            for n in range(20):
                random_inargs = [np.random.random(e.shape) for e in sx_inputs]
                res_ad = f_ground_truth(*random_inargs)
                res_gen = casadi_func(*random_inargs)
                if not isinstance(res_ad, tuple):
                    res_ad = (res_ad,)
                    res_gen = (res_gen,)
                for i, s in enumerate(sx_inputs):
                    if s.name not in excluded:
                        check_res_inf[i] = max(np.max(np.abs(res_ad[i] - res_gen[i])), check_res_inf[i])
            print(f"{func_name} check inf residual: ")
            for i, s in enumerate(sx_inputs):
                print(f"\t{s.name}:\t", check_res_inf[i])

        # Step 2: Generate raw C code
        cgen = cs.CodeGenerator("raw")
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

        # make the mapping of ccs indices to (i, j)
        ij_pairs_all = []
        for io in [sx_inputs, sx_outputs]:
            for x in io:
                ij_pairs_all.append(
                    ccs_index_to_ij(
                        x.shape[0],
                        x.shape[1],
                        x.sparsity().row(),
                        x.sparsity().colind(),
                    )
                )
        def simplify_conditional(code_str):
            pattern = r"arg\[(\d+)\]\? ([^:;]+) : 0;"
            return re.sub(pattern, r"\2;", code_str)

        def simplify_if(code_str):
            pattern = r"if\s*\(res\[\d+\]!=0\)\s*\s*(.+);"
            return re.sub(pattern, r"\1;", code_str)

        def make_input_ref_access(arg_idx, index):
            i, j = ij_pairs_all[arg_idx][index]
            assert j == 0
            return f"inputs[{arg_idx}]({i})"

        def make_output_ref_access(arg_idx, index):
            out = ""
            if is_hessian:
                row = arg_idx // n_in
                col = arg_idx % n_in
                i, j = ij_pairs_all[arg_idx + n_in][index]
                out = f"outputs[{row}][{col}]({i},{j})"
            else:
                i, j = ij_pairs_all[arg_idx + n_in][index]
                if mat_output:
                    out = f"outputs[{arg_idx}]({i},{j})"
                elif vec_out:
                    out = f"outputs({i})"
            if append:
                out += "+"
            return out

        # Step 4: Replace all casadi generated arrays with Eigen::Ref notation
        processed_lines = []
        func_found = False
        func_done = False
        vec_out = sx_outputs[0].shape[1] == 1
        mat_output = not vec_out
        is_hessian = func_name.endswith("_hess")
        for line in lines:
            # preseve casadi built-in functions like casadi_sq (square)
            if "casadi_real casadi_" in line:
                processed_lines.append(line)

            if not func_found:
                if "static int casadi_f0" in line:
                    processed_lines.append(f"CASADI_SYMBOL_EXPORT void {func_name}(\n")
                    processed_lines.append("  std::vector<Eigen::Ref<Eigen::VectorXd>>& inputs,\n")
                    if mat_output:
                        if is_hessian:
                            processed_lines.append(
                                "  std::vector<std::vector<Eigen::Ref<Eigen::MatrixXd>>>& outputs) {\n"
                            )
                        else:
                            processed_lines.append("  std::vector<Eigen::Ref<Eigen::MatrixXd>>& outputs) {\n")
                    elif sx_outputs[0].shape[1] == 1:
                        processed_lines.append("  Eigen::Ref<Eigen::VectorXd> outputs) {\n")

                    # else:
                    #     processed_lines.append("    Eigen::Ref<Eigen::MatrixXd> outputs) {\n")
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

            # Replace output array access: res[i][index] -> outputs[i](r,c) or output[i,j](r,c)
            line = re.sub(r"res\[(\d+)\]\[(\d+)\]", lambda m: make_output_ref_access(int(m[1]), int(m[2])), line)

            processed_lines.append(line)

        # Step 5: Write new C++ file with Eigen interface
        final_cpp_path = os.path.join(output_dir, f"{func_name}.cpp")
        with open(final_cpp_path, "w") as f:
            f.write("#include <vector>\n#include <Eigen/Dense>\n\n")
            f.write("#define casadi_real double\n\n")
            f.write("#ifdef __cplusplus\n")
            # f.write("#ifndef CASADI_SYMBOL_EXPORT\n")
            f.write("#if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)\n")
            f.write("    #if defined(STATIC_LINKED)\n")
            f.write("    #define CASADI_SYMBOL_EXPORT\n")
            f.write("    #else\n")
            f.write("    #define CASADI_SYMBOL_EXPORT __declspec(dllexport)\n")
            f.write("    #endif\n")
            f.write("#elif defined(__GNUC__)\n")
            f.write('    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))\n')
            f.write("#endif\n")
            # f.write("#endif\n")
            f.write('extern "C" {\n')
            f.write("#endif\n\n")
            f.writelines(processed_lines)
            f.write("\n\n#ifdef __cplusplus\n")
            f.write("}\n")
            f.write("#endif\n\n")

        print(f"Generated: {final_cpp_path}")
        make_func_json(func_name, sx_inputs, sx_outputs, output_dir)
        if compile:
            # Step 6: Compile the generated C++ code into a shared library
            so_file_path = os.path.join(output_dir, f"lib{func_name}.so")
            compile_command = f"g++ -shared -fPIC -O3 -DNDEBUG -std=gnu++20 -march=native -o {so_file_path} {final_cpp_path} -I /usr/include/eigen3"
            os.system(compile_command)
            print(f"Compiled:  {so_file_path}")
            if not keep_raw:
                os.remove(cpp_path + "raw.c")
                print(f"Removed:   {cpp_path + 'raw.c'}")
            if not keep_c_src:
                os.remove(final_cpp_path)
                print(f"Removed:   {final_cpp_path}")

    for name, inputs, outputs, kwargs_ in worker:
        process_.append(
            Process(
                target=impl,
                args=(name, inputs, outputs),
                kwargs=kwargs_,
            )
        )
        process_[-1].start()


def wait_until_generated():
    """
    must call this to ensure all tests are done
    """
    for p in process_:
        p.join()
    process_.clear()
    print("All code generation completed.")
