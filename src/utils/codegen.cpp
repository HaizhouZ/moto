#include <moto/utils/codegen.hpp>

namespace moto {
namespace utils {

// Namespace aliases
namespace fs = std::filesystem;
using json = nlohmann::json;
namespace cs_codegen {
namespace impl {
worker_list workers_{};
std::mutex mutex_{};
// Generates a list of (row, col) pairs from CasADi's CCS sparsity format
std::vector<std::pair<int, int>> ccs_index_to_ij(const cs::Sparsity &sp) {
    std::vector<std::pair<int, int>> ij_pairs;
    const auto *row_indices = sp.row();
    const auto *col_offsets = sp.colind();
    for (int j = 0; j < sp.columns(); ++j) {
        for (int k = col_offsets[j]; k < col_offsets[j + 1]; ++k) {
            ij_pairs.emplace_back(row_indices[k], j);
        }
    }
    return ij_pairs;
}
// Transforms raw C code to modern C++ with Eigen
std::string process_generated_code(
    const std::string &raw_c_code,
    const std::string &func_name,
    const std::vector<cs::SX> &sx_inputs,
    const std::vector<cs::SX> &sx_outputs,
    bool append) {
    bool is_hessian = func_name.find("_hess") != std::string::npos;
    // Pre-compute CCS to (row, col) index maps
    std::vector<std::vector<std::pair<int, int>>> ij_pairs_all;
    for (const auto &x : sx_inputs) {
        ij_pairs_all.push_back(ccs_index_to_ij(x.sparsity()));
    }
    for (const auto &y : sx_outputs) {
        ij_pairs_all.push_back(ccs_index_to_ij(y.sparsity()));
    }
    size_t n_in = sx_inputs.size();

    bool vec_out = sx_outputs.size() == 1 && sx_outputs[0].is_column() && !is_hessian;

    // Lambda for generating replacement strings
    auto make_input_ref_access = [&](int arg_idx, int index) -> std::string {
        auto [i, j] = ij_pairs_all.at(arg_idx).at(index);
        return "inputs[" + std::to_string(arg_idx) + "](" + std::to_string(i) + ")";
    };
    auto make_output_ref_access = [&](int arg_idx, int index) -> std::string {
        auto [i, j] = ij_pairs_all.at(n_in + arg_idx).at(index);
        std::string out;
        if (is_hessian) {
            size_t row = arg_idx / n_in;
            size_t col = arg_idx % n_in;
            out = fmt::format("outputs[{}][{}]({},{})", row, col, i, j);
        } else if (vec_out) {
            out = fmt::format("outputs({})", i);
        } else {
            out = fmt::format("outputs[{}]({},{})", arg_idx, i, j);
        }
        if (append)
            out += "+";
        return out;
    };

    std::stringstream processed_code;
    processed_code << "#include <vector>\n#include <Eigen/Dense>\n\n"
                   << "#define casadi_real double\n\n"
                   << "#if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)\n"
                   << "    #define CASADI_SYMBOL_EXPORT __declspec(dllexport)\n"
                   << "#elif defined(__GNUC__)\n"
                   << "    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility (\"default\")))\n"
                   << "#endif\n\n"
                   << "extern \"C\" {\n\n";

    std::stringstream raw_stream(raw_c_code);
    std::string line;
    bool func_found = false, func_done = false;

    // RE2 patterns
    RE2 simplify_cond_re("arg\\[\\d+\\]\\? ([^:;]+) : 0;");
    RE2 simplify_if_re("if\\s*\\(res\\[\\d+\\]!=0\\)\\s*\\s*(.+);");
    RE2 arg_re("arg\\[(\\d+)\\]\\[(\\d+)\\]");
    RE2 res_re("res\\[(\\d+)\\]\\[(\\d+)\\]");

    while (std::getline(raw_stream, line)) {
        if (line.find("casadi_real casadi_") != std::string::npos) {
            processed_code << line << "\n";
        }

        if (!func_found) {
            if (line.find("static int casadi_f0") != std::string::npos) {
                processed_code << "CASADI_SYMBOL_EXPORT void " << func_name << "(\n"
                               << "  std::vector<Eigen::Ref<Eigen::VectorXd>>& inputs,\n";
                if (vec_out) {
                    processed_code << "  Eigen::Ref<Eigen::VectorXd> outputs) {\n";
                } else {
                    if (is_hessian) {
                        processed_code << "  std::vector<std::vector<Eigen::Ref<Eigen::MatrixXd>>>& outputs) {\n";
                    } else {
                        processed_code << "  std::vector<Eigen::Ref<Eigen::MatrixXd>>& outputs) {\n";
                    }
                }
                func_found = true;
            }
            continue;
        }

        if (line.find("return 0;") != std::string::npos)
            continue;
        if (line.find("}") != std::string::npos && !func_done) {
            processed_code << "}\n";
            func_done = true;
            break;
        }

        // Apply RE2 transformations
        RE2::GlobalReplace(&line, simplify_cond_re, "\\1;");
        RE2::GlobalReplace(&line, simplify_if_re, "\\1;");

        // Iteratively replace arg and res patterns
        int cap1, cap2;
        while (RE2::PartialMatch(line, arg_re, &cap1, &cap2)) {
            RE2::Replace(&line, "arg\\[" + std::to_string(cap1) + "\\]\\[" + std::to_string(cap2) + "\\]", make_input_ref_access(cap1, cap2));
        }
        while (RE2::PartialMatch(line, res_re, &cap1, &cap2)) {
            RE2::Replace(&line, "res\\[" + std::to_string(cap1) + "\\]\\[" + std::to_string(cap2) + "\\]", make_output_ref_access(cap1, cap2));
        }

        processed_code << line << "\n";
    }

    processed_code << "\n} // extern \"C\"\n";
    return processed_code.str();
}

// Replicates Python's filter_func_near_zero
cs::Function filter_func_near_zero(
    const std::string &func_name,
    const task::in_arg_list_t &inputs,
    const std::vector<cs::SX> &expr,
    double tol = 1e-8,
    int ntrials = 50) {
    cs::SXVector sx_inputs;
    for (const sym &i : inputs)
        sx_inputs.push_back(i);
    cs::SXVector sx_expr;
    for (const auto &e : expr)
        sx_expr.push_back(e);

    cs::Function f(func_name, sx_inputs, sx_expr);
    std::vector<cs::DM> nz_cnt;
    nz_cnt.reserve(expr.size());
    for (const auto &e : expr) {
        nz_cnt.push_back(cs::DM::zeros(e.rows(), e.columns()));
    }

    for (int n = 0; n < ntrials; ++n) {
        std::vector<cs::DM> inputs_data;
        for (const auto &s : sx_inputs) {
            inputs_data.push_back(2 * (cs::DM::rand(s.rows(), s.columns()) - 0.5));
        }
        auto res = f(inputs_data);
        for (size_t k = 0; k < res.size(); ++k) {
            for (int i = 0; i < res[k].rows(); ++i) {
                for (int j = 0; j < res[k].columns(); ++j) {
                    if (std::abs(static_cast<double>(res[k](i, j))) > tol) {
                        nz_cnt[k](i, j) += 1;
                    }
                }
            }
        }
    }

    cs::SXVector filtered_expr;
    for (size_t k = 0; k < expr.size(); ++k) {
        cs::SX filtered = cs::SX::zeros(cs::Sparsity(expr[k].rows(), expr[k].columns()));
        for (int i = 0; i < nz_cnt[k].rows(); ++i) {
            for (int j = 0; j < nz_cnt[k].columns(); ++j) {
                if (std::round(nz_cnt[k](i, j).scalar()) == ntrials) {
                    filtered(i, j) = expr[k](i, j);
                }
            }
        }
        filtered_expr.push_back(filtered);
    }

    return cs::Function(func_name, sx_inputs, filtered_expr);
}

// Core implementation logic for a single function
void run(
    std::string func_name,
    task::in_arg_list_t sx_inputs,
    std::vector<cs::SX> sx_outputs,
    std::string output_dir,
    std::string compile_flag,
    bool force_recompile,
    bool append,
    cs::Function func_ground_truth,
    bool keep_generated_src,
    bool verbose) {
    // Step 1: Create CasADi function and filter near-zero elements
    cs::Function casadi_func = filter_func_near_zero(func_name, sx_inputs, sx_outputs);
    std::vector<cs::SX> sx_inputs_cs; //(sx_inputs.begin(), sx_inputs.end());
    for (cs::SX &s : sx_inputs) {
        sx_inputs_cs.emplace_back(s);
    }
    auto filtered_outputs = casadi_func(sx_inputs_cs);

    // Step 2: Generate raw C code
    cs::CodeGenerator cgen(func_name + "_raw.c");
    cgen.add(casadi_func);
    fs::create_directories(output_dir);
    std::string raw_c_path = fs::path(output_dir) / (func_name + "_raw.c");
    cgen.generate(output_dir + '/'); // Generates file in the specified dir

    // Step 3: Parse raw C code and transform it
    std::ifstream raw_file(raw_c_path);
    std::stringstream buffer;
    buffer << raw_file.rdbuf();
    raw_file.close();

    std::string processed_code = process_generated_code(
        buffer.str(), func_name, sx_inputs_cs, filtered_outputs, append);

    // Step 4: Write new C++ file with Eigen interface
    std::string final_cpp_path = fs::path(output_dir) / (func_name + ".cpp");
    std::ofstream final_cpp_file(final_cpp_path);
    final_cpp_file << processed_code;
    final_cpp_file.close();
    if (verbose)
        std::cout << "Generated: " << final_cpp_path << std::endl;

    // Step 5: Compile if necessary
    fs::path so_file_path = fs::path(output_dir) / ("lib" + func_name + ".so");
    fs::path json_path = fs::path(output_dir) / (func_name + ".json");
    std::string md5_hash = compute_md5(final_cpp_path);

    bool needs_compile = true;
    bool json_exists = fs::exists(json_path);
    bool so_exists = fs::exists(so_file_path);
    json data;
    if (!force_recompile && json_exists && so_exists) {
        std::ifstream jf(json_path);
        data = json::parse(jf);
        if (data["md5"] == md5_hash && data["compile_flag"] == compile_flag) {
            if (verbose)
                std::cout << "Skipping " << func_name << " as it is already up-to-date." << std::endl;
            needs_compile = false;
        }
    }

    if (needs_compile) {
        std::string eigen_include_path = "/usr/include/eigen3"; // Adjust if necessary
        std::string compile_command = "g++ -shared -fPIC -std=c++20 " + compile_flag +
                                      " -o " + so_file_path.string() + " " + final_cpp_path +
                                      " -I " + eigen_include_path;
        int ret = std::system(compile_command.c_str());
        if (verbose) {
            if (ret == 0) {
                std::cout << "Compiled:  " << so_file_path << std::endl;
            } else {
                std::cerr << "Compilation failed for: " << func_name << std::endl;
            }
        }
    }

    // Step 6: Create JSON metadata and cleanup
    fs::create_directories(output_dir);
    json j;
    j["name"] = func_name;
    for (const sym &e : sx_inputs) {
        j["inputs"][e.name()] = {e.dim(), static_cast<int>(e.field())};
    }
    for (const auto &e : sx_outputs) {
        j["outputs"].push_back({e.rows(), e.columns()});
    }
    j["md5"] = md5_hash;
    j["compile_flag"] = compile_flag;

    std::ofstream o(fs::path(output_dir) / (func_name + ".json"));
    o << std::setw(4) << j << std::endl;
    if (!keep_generated_src) {
        fs::remove(raw_c_path);
        fs::remove(final_cpp_path);
    }
}

}; // namespace impl

void task::finalize(worker_list &workers_) {
    std::string full_func_name = prefix.empty() ? func_name : prefix + "_" + func_name;

    if (gen_eval)
        workers_.add(std::async(std::launch::deferred,
                                &impl::run,
                                full_func_name,
                                sx_inputs,
                                std::vector{sx_output},
                                output_dir,
                                eval_compile_flag,
                                force_recompile,
                                append_value, // 'append' flag
                                cs::Function(),
                                keep_generated_src,
                                verbose));

    // excluded = [e.name for e in exclude]
    std::set<size_t> excluded;
    // # exclude inputs in field p
    for (const sym &s : sx_inputs)
        if (s.field() == __p)
            excluded.insert(s.uid());
    std::map<size_t, cs::SX> external_jac;
    for (auto &[in_arg, jac] : ext_jac)
        external_jac[in_arg->uid()] = std::move(jac);

    std::map<std::pair<size_t, size_t>, cs::SX> external_hess;
    for (auto &[in_arg0, in_arg1, hess] : ext_hess) {
        size_t uid0 = in_arg0->uid();
        size_t uid1 = in_arg1->uid();
        if (uid0 < uid1) {
            external_hess[{uid0, uid1}] = std::move(hess);
        } else {
            external_hess[{uid1, uid0}] = std::move(hess);
        }
    }
    if (!ext_jac.empty())
        gen_jacobian = true;
    if (!ext_hess.empty())
        gen_hessian = true;
    bool vjp_for_hess = false;

    assert(sx_output.columns() == 1);

    if (sx_output.rows() > 1 && ext_hess.empty() && gen_hessian) {
        // will use jacobian to compute vjp->hessian
        vjp_for_hess = true;
    }

    std::vector<cs::SX> jacs;
    std::vector<cs::SX> jacs_copy;
    // generate jacobian
    if (gen_jacobian or vjp_for_hess) {
        for (sym &s : sx_inputs) {
            if (!excluded.contains(s.uid())) {
                if (!external_jac.empty() and external_jac.contains(s.uid())) {
                    jacs.push_back(external_jac[s.uid()]);
                    continue;
                }
                jacs.push_back(cs::SX::jacobian(sx_output, s));
            } else
                jacs.push_back(cs::SX());
        }
        if (gen_jacobian and !jacs.empty()) {
            cs::Function f_ad;
            if (check_jac_ad) {
                std::vector<cs::SX> sx_inputs_cs;
                std::vector<cs::SX> jac_ad;
                for (sym &e : sx_inputs) {
                    sx_inputs_cs.push_back(e);
                    jac_ad.push_back(cs::SX::jacobian(sx_output, e));
                }
                f_ad = cs::Function(func_name + "_ad", sx_inputs_cs, jac_ad);
            }
            if (gen_hessian && !vjp_for_hess) {
                // if we are generating hessian, we need to copy jacs
                if (gen_jacobian)
                    jacs_copy = jacs;
                else
                    jacs_copy = std::move(jacs);
            }
            workers_.add(std::async(std::launch::deferred,
                                    &impl::run,
                                    full_func_name + "_jac",
                                    sx_inputs,
                                    std::move(jacs),
                                    output_dir,
                                    jac_compile_flag,
                                    force_recompile,
                                    append_jac, // 'append' flag
                                    f_ad,
                                    keep_generated_src,
                                    verbose));
        }
    }
    // generate hessian
    std::vector<std::vector<cs::SX>> hess;
    task::in_arg_list_t hess_inputs;
    if (gen_hessian) {
        hess.resize(sx_inputs.size());
        // use AD of vjp to compute hessian

        auto lbd = vjp_for_hess ? sym(func_name + "_lbd", sx_output.rows(), __usr_var) : sym();
        for (size_t idx_i = 0; idx_i < sx_inputs.size(); ++idx_i) {
            sym &i = sx_inputs[idx_i];
            hess[idx_i].resize(sx_inputs.size());
            cs::SX vjp_;
            bool i_excluded = excluded.contains(i.uid());
            if (vjp_for_hess && !i_excluded)
                vjp_ = cs::SX::jtimes(sx_output, i, lbd, true);
            size_t idx_j = 0;
            for (size_t idx_j = 0; idx_j < sx_inputs.size(); ++idx_j) {
                sym &j = sx_inputs[idx_j];
                if (i_excluded or excluded.contains(j.uid()) or i.field() < j.field()) {
                    continue;
                }
                if (!vjp_for_hess) {
                    if (external_hess.contains({i.uid(), j.uid()})) {
                        hess[idx_i][idx_j] = external_hess[{i.uid(), j.uid()}];
                        continue;
                    } else if (external_hess.contains({j.uid(), i.uid()})) {
                        hess[idx_i][idx_j] = external_hess[{j.uid(), i.uid()}].T();
                        continue;
                    }
                } else if (i.field() == j.field() and idx_i > idx_j) {
                    // for i,j in same field, just copy
                    hess[idx_i][idx_j] = hess[idx_j][idx_i].T();
                    continue;
                }
                if (vjp_for_hess) {
                    hess[idx_i][idx_j] = cs::SX::jacobian(vjp_, j);
                } else {
                    hess[idx_i][idx_j] = cs::SX::jacobian(jacs_copy[idx_i], j);
                }
            }
            hess_inputs.reserve(sx_inputs.size() + 1);
            hess_inputs = sx_inputs;
            if (vjp_for_hess) {
                hess_inputs.push_back(lbd);
            }
        }
        // hess = [item for sublist in hess for item in sublist]
        std::vector<cs::SX> hess_flat;
        hess_flat.reserve(sx_inputs.size() * sx_inputs.size());
        for (auto &sublist : hess) {
            hess_flat.insert(hess_flat.end(), sublist.begin(), sublist.end());
        }
        workers_.add(std::async(std::launch::deferred,
                                &impl::run,
                                full_func_name + "_hess",
                                hess_inputs,
                                std::move(hess_flat),
                                output_dir,
                                hess_compile_flag,
                                force_recompile,
                                true, // 'append' flag
                                cs::Function(),
                                keep_generated_src,
                                verbose));
    }
}
// Public entry point to start code generation
worker_list generate_and_compile(task &&_task) {
    worker_list workers_tmp;
    _task.finalize(workers_tmp);
    std::lock_guard<std::mutex> lock(impl::mutex_);
    impl::workers_.add(workers_tmp);
    // Launch the implementation in a new thread
    return workers_tmp;
}

// Waits for all compilation threads to finish
void wait_until_generated() {
    std::lock_guard<std::mutex> lock(impl::mutex_);
    std::cout << "Waiting for code generation tasks to complete..." << std::endl;
    impl::workers_.wait_until_finished();
    impl::workers_.workers.clear();
    std::cout << "All code generation completed." << std::endl;
}
} // namespace cs_codegen
} // namespace utils
} // namespace moto