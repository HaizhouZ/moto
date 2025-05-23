#ifndef __ATRI_CODE_GEN__
#define __ATRI_CODE_GEN__

#include <atri/core/expr.hpp>
#include <casadi/casadi.hpp>

namespace atri {

namespace cs = casadi;

struct codegen_opts {
    /// sx_inputs Vector of CasADi SX inputs.
    std::vector<sym> sx_inputs;
    /// output_dir Directory to write generated files.
    std::string output_dir = "gen";
    /// compile If true, compile the generated C++ code to a shared library.
    bool compile = true;
    /// keep_raw If false, remove raw generated C code after compilation.
    bool keep_raw = false;
    /// keep_c_src If false, remove generated C++ source after compilation.
    bool keep_c_src = false;
    /// list of excluded symbols
    std::vector<sym> exclude = {};
    /// generate zero order evaluation
    bool gen_eval = true;
    /// generate jacobian
    bool gen_jacobian = false;
    /// generate hessian
    bool gen_hessian = false;
    /// external jacobian [(input, jacobian)]
    std::vector<std::pair<sym, cs::SX>> ext_jac = {};
    /// check the jacobian with auto-differentiation
    bool check_jac_ad = false;
    /// [input1, input2, jacobian)]
    std::vector<std::tuple<sym, sym, cs::SX>> ext_hess = {};
    /// (not implemented) check the hessian with auto-differentiation
    bool check_hess_ad = false;
};

/**
 * @brief generate casadi code compatible with eigen and compile
 * Generate Eigen-compatible C++ code from CasADi SX expressions.
 * This function creates a CasADi function, filters zeros, generates C code,
 * parses and rewrites it to use Eigen::Ref, and optionally compiles it.
 * @param func_name Name of the function to generate.
 */
void generate_and_compile(const std::string &func_name,
                          codegen_opts opt,
                          cs::SX sx_output);

void wait_until_generated();
} // namespace atri
#endif