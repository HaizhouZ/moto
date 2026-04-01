#include <moto/utils/codegen.hpp>

namespace moto {
namespace utils {

// Namespace aliases
namespace fs = std::filesystem;
using json = nlohmann::json;
namespace cs_codegen {

void job_list::wait_until_finished() {
    for (auto &w : jobs) {
        w();
    }
}

auto mat_type() {
    if constexpr (std::is_same_v<scalar_t, double>) {
        return "MatrixXd";
    } else {
        return "MatrixXf";
    }
}

auto vec_type() {
    if constexpr (std::is_same_v<scalar_t, double>) {
        return "VectorXd";
    } else {
        return "VectorXf";
    }
}

namespace impl {
// job_list jobs_{};
std::mutex mutex_{};
std::mutex func_mutex_map_mutex_{};
std::unordered_map<std::string, std::shared_ptr<std::mutex>> func_mutexes_{};
std::unordered_map<std::string, std::string> completed_compile_flags_{};

std::shared_ptr<std::mutex> get_func_mutex(const std::string &func_name) {
    std::lock_guard<std::mutex> lock(func_mutex_map_mutex_);
    auto [it, inserted] = func_mutexes_.try_emplace(func_name, std::make_shared<std::mutex>());
    return it->second;
}
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
    bool append,
    bool with_aux) {
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

    bool is_jac = func_name.find("_jac") != std::string::npos;

    bool vec_out = !is_jac && !is_hessian;

    // Lambda for generating replacement strings
    auto make_input_ref_access = [&](int arg_idx, int index) -> std::string {
        auto [i, j] = ij_pairs_all.at(arg_idx).at(index);
        ///@todo
        std::stringstream out;
        out << fmt::format("inputs[{}].data() ? ", arg_idx);
        out << fmt::format("inputs[{}]({}) : 0", arg_idx, i);
        return out.str();
    };
    auto make_output_ref_access = [&](int arg_idx, int index) -> std::string {
        auto [i, j] = ij_pairs_all.at(n_in + arg_idx).at(index);
        std::stringstream out;
        if (is_hessian) {
            size_t row = arg_idx / (n_in - with_aux);
            size_t col = arg_idx % (n_in - with_aux);
            out << fmt::format("if (outputs[{}][{}].data()) ", row, col);
            out << fmt::format("outputs[{}][{}]({},{})", row, col, i, j);
        } else if (vec_out) {
            out << fmt::format("outputs({})", i);
        } else {
            out << fmt::format("if (outputs[{}].data()) ", arg_idx);
            out << fmt::format("outputs[{}]({},{})", arg_idx, i, j);
        }
        if (append)
            out << "+";
        return out.str();
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
                               << "  std::vector<Eigen::Ref<Eigen::" << vec_type() << ">>& inputs,\n";
                if (vec_out) {
                    processed_code << "  Eigen::Ref<Eigen::" << vec_type() << "> outputs) {\n";
                } else {
                    if (is_hessian) {
                        processed_code << "  std::vector<std::vector<Eigen::Ref<Eigen::" << mat_type() << ">>>& outputs) {\n";
                    } else {
                        processed_code << "  std::vector<Eigen::Ref<Eigen::" << mat_type() << ">>& outputs) {\n";
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
    bool verbose,
    cs::SX aux) {
    // Finalized clones intentionally share a stable generated symbol name.
    // Serialize codegen per symbol to avoid concurrent writers racing on
    // func_name_raw.c / func_name.cpp / func_name.json / libfunc_name.so.
    auto func_mutex = get_func_mutex(func_name);
    std::lock_guard<std::mutex> func_lock(*func_mutex);

    fs::path so_file_path = fs::path(output_dir) / ("lib" + func_name + ".so");
    fs::path json_path = fs::path(output_dir) / (func_name + ".json");
    const std::string cache_key = (fs::path(output_dir) / func_name).string();

    if (!force_recompile) {
        std::lock_guard<std::mutex> lock(func_mutex_map_mutex_);
        auto it = completed_compile_flags_.find(cache_key);
        if (it != completed_compile_flags_.end() && it->second == compile_flag) {
            if (std::getenv("MOTO_DEBUG_CODEGEN") != nullptr) {
                fmt::print("[codegen] reuse {}\n", func_name);
            }
            return;
        }
    }

    // Step 1: Create CasADi function and filter near-zero elements
    std::vector<cs::SX> sx_inputs_cs; //(sx_inputs.begin(), sx_inputs.end());
    sx_inputs_cs.reserve(sx_inputs.size() + !aux.is_empty());
    for (cs::SX &s : sx_inputs) {
        sx_inputs_cs.emplace_back(s);
    }
    if (!aux.is_empty()) {
        // throw std::runtime_error("Auxiliary variable is not supported in this context.");
        sx_inputs_cs.emplace_back(aux);
    }
    // auto filtered_outputs = casadi_func(sx_inputs_cs);
    cs::SXVector filtered_outputs;
    for (auto &e : sx_outputs)
        filtered_outputs.push_back(cs::SX::sparsify(e));

    auto casadi_func = cs::Function(func_name, sx_inputs_cs, filtered_outputs);

    // Step 2: Generate raw C code
    std::string casadi_real_t = std::is_same_v<scalar_t, double> ? "double" : "float";
    cs::Dict opts;
    opts["casadi_real"] = casadi_real_t;
    cs::CodeGenerator cgen(func_name + "_raw.c", opts);
    cgen.add(casadi_func);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        fs::create_directories(output_dir);
    }
    std::string raw_c_path = fs::path(output_dir) / (func_name + "_raw.c");
    cgen.generate(output_dir + '/'); // Generates file in the specified dir

    // Step 3: Parse raw C code and transform it
    std::stringstream buffer;
    {
        std::ifstream raw_file(raw_c_path);
        buffer << raw_file.rdbuf();
    }
    std::string raw_c_code = buffer.str();

    std::string processed_code = process_generated_code(
        raw_c_code, func_name, sx_inputs_cs, filtered_outputs, append, !aux.is_empty());

    // Step 4: Write new C++ file with Eigen interface
    std::string final_cpp_path = fs::path(output_dir) / (func_name + ".cpp");
    {
        std::ofstream final_cpp_file(final_cpp_path);
        final_cpp_file << processed_code;
    }
    if (verbose)
        std::cout << "Generated: " << final_cpp_path << std::endl;

    // Step 5: Compile if necessary
    std::string md5_hash = compute_md5_from_bytes(raw_c_code) + compute_md5_from_bytes(processed_code);

    bool needs_compile = true;
    bool json_exists = fs::exists(json_path);
    bool so_exists = fs::exists(so_file_path);
    json data;
    if (!force_recompile && json_exists && so_exists) {
        try {
            {
                std::ifstream jf(json_path);
                data = json::parse(jf);
            }
            if (data["md5"] == md5_hash && data["compile_flag"] == compile_flag) {
                if (verbose)
                    std::cout << "Skipping " << func_name << " as it is already up-to-date." << std::endl;
                needs_compile = false;
            } else {
                fmt::print("Recompiling {}: md5 mismatch or compile flag changed.\n", func_name);
                fmt::print("  Current md5: {}, compile flag: {}\n", md5_hash, compile_flag);
                fmt::print("  Previous md5: {}, compile flag: {}\n", std::string(data["md5"]), std::string(data["compile_flag"]));
            }
        } catch (const json::parse_error &e) {
            fmt::print("Error parsing JSON for {}: {}\n", func_name, e.what());
            needs_compile = true;
        }
    }

    if (std::getenv("MOTO_DEBUG_CODEGEN") != nullptr) {
        if (needs_compile) {
            fmt::print("[codegen] compile {}\n", func_name);
            fmt::print("  json_exists={} so_exists={} force_recompile={}\n",
                       json_exists, so_exists, force_recompile);
            fmt::print("  md5_current={} compile_flag_current={}\n", md5_hash, compile_flag);
            if (json_exists && so_exists && !data.is_discarded() && !data.empty()) {
                std::string md5_prev = data.contains("md5") ? std::string(data["md5"]) : "<missing>";
                std::string flag_prev = data.contains("compile_flag") ? std::string(data["compile_flag"]) : "<missing>";
                fmt::print("  md5_prev={} compile_flag_prev={}\n", md5_prev, flag_prev);
            }
        } else {
            fmt::print("[codegen] reuse {}\n", func_name);
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

        // Step 6: Create JSON metadata and cleanup
        json j; /// @todo the order here is not guaranteed
        j["name"] = func_name;
        for (const sym &e : sx_inputs) {
            j["inputs"][e.name()] = {e.dim(), static_cast<int>(e.field())};
        }
        for (const auto &e : filtered_outputs) {
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

    if (!force_recompile) {
        std::lock_guard<std::mutex> lock(func_mutex_map_mutex_);
        completed_compile_flags_[cache_key] = compile_flag;
    }
}

}; // namespace impl

void task::finalize(job_list &jobs_) {
    std::string full_func_name = prefix.empty() ? func_name : prefix + "_" + func_name;
    if (gen_eval)
        jobs_.add(std::bind(&impl::run,
                            full_func_name,
                            sx_inputs,
                            std::vector{!gauss_newton ? sx_output : 0.5 * cs::SX::dot(sx_output, sx_output * weight_gn)},
                            output_dir,
                            eval_compile_flag,
                            force_recompile,
                            append_value, // 'append' flag
                            cs::Function(),
                            keep_generated_src,
                            verbose,
                            cs::SX()));

    // excluded = [e.name for e in exclude]
    std::set<size_t> excluded;
    // exclude non-primal storage-only inputs
    for (const sym &s : sx_inputs)
        if (s.field() == __p || s.field() == __s)
            excluded.insert(s.uid());
    std::map<size_t, cs::SX> external_jac;
    for (auto &[in_arg, jac] : ext_jac) {
        if (jac.columns() != in_arg->tdim() || jac.rows() != sx_output.rows())
            throw std::runtime_error(fmt::format("Jacobian dimension mismatch for sym {} in field {}, expected ({}, {}), got ({}, {})",
                                                 in_arg->name(), in_arg->field(), sx_output.rows(), in_arg->tdim(), jac.rows(), jac.columns()));
        external_jac[in_arg->uid()] = std::move(jac);
    }

    std::map<std::pair<size_t, size_t>, cs::SX> external_hess;
    for (auto &[in_arg0, in_arg1, hess] : ext_hess) {
        size_t uid0 = in_arg0->uid();
        size_t uid1 = in_arg1->uid();
        if (uid0 < uid1) {
            if (hess.rows() != in_arg0->tdim() || hess.columns() != in_arg1->tdim())
                throw std::runtime_error(fmt::format("Hessian dimension mismatch for syms {} (field {}) and {} (field {}), expected ({}, {}), got ({}, {})",
                                                     in_arg0->name(), in_arg0->field(), in_arg1->name(), in_arg1->field(),
                                                     in_arg0->tdim(), in_arg1->tdim(), hess.rows(), hess.columns()));
            external_hess[{uid0, uid1}] = std::move(hess);
        } else {
            if (hess.rows() != in_arg1->tdim() || hess.columns() != in_arg0->tdim())
                throw std::runtime_error(fmt::format("Hessian dimension mismatch for syms {} (field {}) and {} (field {}), expected ({}, {}), got ({}, {})",
                                                     in_arg1->name(), in_arg1->field(), in_arg0->name(), in_arg0->field(),
                                                     in_arg1->tdim(), in_arg0->tdim(), hess.rows(), hess.columns()));
            external_hess[{uid1, uid0}] = hess.T();
        }
    }
    if (!ext_jac.empty())
        gen_jacobian = true;
    if (!ext_hess.empty())
        gen_hessian = true;
    bool merit_jac_for_hess = false; /// true if we need to use jacobian to compute vjp->hessian

    assert(sx_output.columns() == 1);

    if (sx_output.rows() > 1 && ext_hess.empty() && gen_hessian && !gauss_newton) {
        // will use jacobian to compute vjp->hessian
        merit_jac_for_hess = true;
    }

    auto get_dstep_ds = [](const sym &s) -> cs::SX {
        /// @todo : this assumes affine dependence on step size, which may not be true for all cases (i.e., hessian wrt step size will be zero)
        /// for example if the integration is s + step ^ 2, the jacobian will contain step which is not an input to the function (is it necessary?)
        auto step = cs::SX::sym(s.name() + "_step", s.tdim());
        return cs::SX::jacobian(s.symbolic_integrate(s, step), step);
    };

    std::vector<cs::SX> jacs;
    std::vector<cs::SX> jacs_copy;
    // generate jacobian
    if (gen_jacobian or merit_jac_for_hess) {
        for (sym &s : sx_inputs) {
            if (!excluded.contains(s.uid())) {
                if (!external_jac.empty() and external_jac.contains(s.uid())) {
                    jacs.push_back(external_jac[s.uid()]);
                    continue;
                }
                if (s.has_non_trivial_integration()) {
                    // get jacobian wrt step (variation)
                    auto j = cs::SX::mtimes(cs::SX::jacobian(sx_output, s), get_dstep_ds(s));
                    jacs.push_back(j);
                } else {
                    jacs.push_back(cs::SX::jacobian(sx_output, s));
                }
            } else
                jacs.push_back(cs::SX());
        }
        if (gen_jacobian and !jacs.empty()) {
            cs::Function f_ad;
            if (!jac_outputs.empty()) {
                jacs = jac_outputs;
                if (gen_hessian)
                    throw std::runtime_error("Cannot compute hessian when multiple jacobian outputs are specified.");
            } else {
                if (check_jac_ad) {
                    /// @warning not applicable to nontrivial integration
                    std::vector<cs::SX> sx_inputs_cs;
                    std::vector<cs::SX> jac_ad;
                    for (sym &e : sx_inputs) {
                        sx_inputs_cs.push_back(e);
                        jac_ad.push_back(cs::SX::jacobian(sx_output, e));
                    }
                    f_ad = cs::Function(func_name + "_ad", sx_inputs_cs, jac_ad);
                }
                if (gen_hessian && !merit_jac_for_hess) {
                    // if we are generating hessian, we need to copy jacs
                    if (gen_jacobian) {
                        jacs_copy = jacs;
                        if (gauss_newton) {
                            for (auto &jac : jacs) {
                                if (jac.is_empty())
                                    continue;
                                jac = cs::SX::mtimes((weight_gn * sx_output).T(), jac);
                            }
                        }
                    } else
                        jacs_copy = std::move(jacs);
                }
            }
            jobs_.add(std::bind(&impl::run,
                                full_func_name + "_jac",
                                sx_inputs,
                                std::move(jacs),
                                output_dir,
                                jac_compile_flag,
                                force_recompile,
                                append_jac, // 'append' flag
                                f_ad,
                                keep_generated_src,
                                verbose, cs::SX()));
        }
    }

    if (gauss_newton) {
        for (size_t i : range(sx_inputs.size())) {
            for (size_t j : range(i, sx_inputs.size())) {
                sym &s = sx_inputs[i];
                sym &t = sx_inputs[j];
                if (excluded.contains(s.uid()) or excluded.contains(t.uid()))
                    continue;
                external_hess[{s.uid(), t.uid()}] = cs::SX::mtimes(jacs_copy[i].T(), cs::SX::mtimes(cs::SX::diag(weight_gn), jacs_copy[j]));
            }
        }
    }

    // generate hessian
    std::vector<std::vector<cs::SX>> hess;
    if (gen_hessian) {
        hess.resize(sx_inputs.size());
        // use AD of vjp to compute hessian if merit_jac_for_hess is true
        auto lbd = merit_jac_for_hess ? cs::SX::sym(func_name + "_lbd", sx_output.rows()) : cs::SX();
        for (size_t idx_i = 0; idx_i < sx_inputs.size(); ++idx_i) {
            sym &i = sx_inputs[idx_i];
            hess[idx_i].resize(sx_inputs.size());
            cs::SX merit_jac_;
            if (excluded.contains(i.uid()))
                continue;
            if (merit_jac_for_hess)
                merit_jac_ = cs::SX::jtimes(sx_output, i, lbd, true);
            size_t idx_j = 0;
            for (size_t idx_j = 0; idx_j < sx_inputs.size(); ++idx_j) {
                sym &j = sx_inputs[idx_j];
                if (excluded.contains(j.uid()) or i.field() < j.field()) {
                    continue;
                }
                if (!merit_jac_for_hess) {
                    if (external_hess.contains({i.uid(), j.uid()})) {
                        hess[idx_i][idx_j] = external_hess[{i.uid(), j.uid()}];
                        goto HESS_SETUP_SPARSITY;
                    } else if (external_hess.contains({j.uid(), i.uid()})) {
                        hess[idx_i][idx_j] = external_hess[{j.uid(), i.uid()}].T();
                        goto HESS_SETUP_SPARSITY;
                    }
                } else if (i.field() == j.field() and idx_i > idx_j) {
                    // for i,j in same field, just copy
                    hess[idx_i][idx_j] = hess[idx_j][idx_i].T();
                    if (hess_sp != nullptr)
                        (*hess_sp)[idx_i][idx_j] = (*hess_sp)[idx_j][idx_i];
                    continue;
                }
                if (merit_jac_for_hess) {
                    hess[idx_i][idx_j] = cs::SX::sparsify(cs::SX::jacobian(merit_jac_, j));
                } else {
                    hess[idx_i][idx_j] = cs::SX::sparsify(cs::SX::jacobian(jacs_copy[idx_i], j));
                }
            HESS_SETUP_SPARSITY:
                if (hess[idx_i][idx_j].is_zero()) {
                    hess[idx_i][idx_j] = cs::SX();
                    if (hess_sp != nullptr)
                        (*hess_sp)[idx_i][idx_j] = sparsity::unknown; // no hessian
                    continue;
                } else if (j.has_non_trivial_integration()) { // apply integration
                    hess[idx_i][idx_j] = cs::SX::mtimes(hess[idx_i][idx_j], get_dstep_ds(j));
                }
                if (hess[idx_i][idx_j].is_square() && hess[idx_i][idx_j].sparsity().is_diag()) {
                    bool is_eye = false;
                    hess[idx_i][idx_j] = cs::SX::diag(hess[idx_i][idx_j]);
                    if (hess[idx_i][idx_j].is_one())
                        is_eye = true;
                    if (is_eye)
                        hess[idx_i][idx_j] = cs::SX::ones(hess[idx_i][idx_j].rows());
                    if (hess_sp != nullptr)
                        (*hess_sp)[idx_i][idx_j] = is_eye ? sparsity::eye : sparsity::diag;
                } else {
                    if (hess_sp != nullptr)
                        (*hess_sp)[idx_i][idx_j] = sparsity::dense;
                }
            }
        }
        // hess = [item for sublist in hess for item in sublist]
        std::vector<cs::SX> hess_flat;
        hess_flat.reserve(sx_inputs.size() * sx_inputs.size());
        for (auto &sublist : hess) {
            hess_flat.insert(hess_flat.end(), sublist.begin(), sublist.end());
        }
        jobs_.add(std::bind(&impl::run,
                            full_func_name + "_hess",
                            sx_inputs,
                            std::move(hess_flat),
                            output_dir,
                            hess_compile_flag,
                            force_recompile,
                            true, // 'append' flag
                            cs::Function(),
                            keep_generated_src,
                            verbose,
                            lbd));
    }
}
// Public entry point to start code generation
job_list generate_and_compile(task &_task) {
    job_list jobs_tmp;
    _task.finalize(jobs_tmp);
    if (_task.extra_task) {
        _task.extra_task->finalize(jobs_tmp);
    }
    // std::lock_guard<std::mutex> lock(impl::mutex_);
    // impl::jobs_.add(jobs_tmp);
    // Launch the implementation in a new thread
    return jobs_tmp;
}

// Waits for all compilation threads to finish
// void wait_until_generated() {
//     std::lock_guard<std::mutex> lock(impl::mutex_);
//     std::cout << "Waiting for code generation tasks to complete..." << std::endl;
//     impl::jobs_.wait_until_finished();
//     impl::jobs_.jobs.clear();
//     std::cout << "All code generation completed." << std::endl;
// }

void server::routine() {
    size_t n_threads = omp_get_max_threads();
    std::mutex thread_mtx_;
    std::condition_variable thread_cv_;
    while (true) {
        std::unique_lock<std::mutex> lock(queue_mtx_);
        queue_cv_.wait(lock, [this] { return !job_buffer_.jobs.empty() || terminated_; });
        if (terminated_) {
            terminated_ = false;
            break; ///< exit the loop if terminated
        }
        auto jobs = std::move(job_buffer_.jobs);
        job_buffer_.jobs.clear();
        lock.unlock();
        for (auto &w : jobs) {
            std::unique_lock<std::mutex> thread_lock(thread_mtx_);
            thread_cv_.wait(thread_lock, [&n_threads] { return n_threads > 0; });
            n_threads--;
            std::thread([&, w = std::move(w)]() mutable {
                w();
                std::lock_guard<std::mutex> thread_lock(thread_mtx_);
                n_threads++;
                thread_cv_.notify_one(); ///< notify the server that the job is done
            }).detach();
        }
    }
    // std::lock_guard<std::mutex> lock(terminate_mtx_);
    // terminate_cv_.notify_one();
} ///< daemon to wait for codegen jobs
} // namespace cs_codegen
} // namespace utils
} // namespace moto
