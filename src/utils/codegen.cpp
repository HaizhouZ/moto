#include <atri/utils/codegen.hpp>
#include <boost/regex.hpp>
#include <filesystem>
#include <future>
#include <thread>

namespace atri {

inline auto make_random(const std::vector<cs::SX> &args) {
    cs::DMVector in_;
    for (auto i : range(args.size())) {
        in_.push_back(cs::DM::rand(args[i].sparsity()));
        in_.back() *= 2.;
        in_.back() -= 1.;
    }
    return in_;
}

inline auto to_sx_list(const std::vector<sym> &inputs) {
    std::vector<cs::SX> out;
    for (auto &s : inputs) {
        assert(s.is_symbolic());
        out.push_back(s);
    }
    return out;
}

/**
 * @brief The nonzeros across all trials will be recorded to generate a new function
 */
inline auto filter_func_near_zero(const std::string &func_name, const std::vector<cs::SX> &inputs, const std::vector<cs::SX> &expr, double tol = 1e-8,
                                  size_t ntrials = 50) {
    cs::Function f(func_name, inputs, expr);
    std::vector<Eigen::MatrixXi> nz_cnt;
    for (auto &e : expr) {
        nz_cnt.push_back(Eigen::MatrixXi(e.rows(), e.columns()));
        nz_cnt.back().setZero();
    }

    for (size_t n : range(ntrials)) {
        cs::DMVector in_;
        for (auto i : range(inputs.size())) {
            in_.push_back(cs::DM::rand(inputs[i].sparsity()));
        }
        auto res = f(in_);
        for (auto k : range(expr.size()))
            for (auto i : range(expr[k].rows()))
                for (auto j : range(expr[k].columns()))
                    // mark if higher than tol
                    if (abs((double)res[k](i, j)) > tol)
                        nz_cnt[k](i, j) += 1;
    }
    std::vector<cs::SX> filtered;
    for (auto &e : expr) {
        filtered.emplace_back(cs::SX::zeros(cs::Sparsity(e.rows(), e.columns())));
    }
    // for each output, find the true nonzeros that passed all trials
    for (size_t k : range(expr.size())) {
        for (size_t i : range(nz_cnt[k].rows())) {
            for (size_t j : range(nz_cnt[k].cols())) {
                if (nz_cnt[k](i, j) == ntrials) {
                    filtered[k](i, j) = expr[k](i, j);
                }
            }
        }
    }

    return cs::Function(func_name, inputs, filtered);
}
/**
 * @brief Create index-ij pair map from CCS format sparsity (default of CasADi)
 */
inline auto ccs_index_to_ij(int rows, int cols, const std::vector<casadi_int> &row, const std::vector<casadi_int> &colind) {
    std::vector<std::pair<int, int>> ij_pairs;
    for (int j = 0; j < cols; ++j) {
        for (int k = colind[j]; k < colind[j + 1]; ++k) {
            int i = row[k];
            ij_pairs.emplace_back(i, j);
        }
    }
    return ij_pairs;
}

/**
 * @brief generate eigen compatible code
 * @param check_required If true, check generated function against ground truth.
 * @param sx_outputs Vector of CasADi SX outputs.
 * @param f_ground_truth Optional ground truth CasADi function for checking.
 */
void generate_eigen_cpp(const std::string &func_name,
                        const codegen_opts &opt,
                        const std::vector<cs::SX> &sx_outputs,
                        bool check_required = false,
                        cs::Function f_ground_truth = {}) {
    // Step 1: Create CasADi function, filter zeros
    auto sx_inputs = to_sx_list(opt.sx_inputs);
    cs::Function casadi_func = filter_func_near_zero(func_name, sx_inputs, sx_outputs);
    std::vector<cs::SX> outputs_vec = casadi_func(sx_inputs);

    if (check_required && f_ground_truth.get() != nullptr) {
        size_t n_in = sx_inputs.size();
        std::vector<double> check_res_inf(n_in, 0.0);
        for (int n = 0; n < 20; ++n) {
            auto random_inargs = make_random(sx_inputs);
            auto res_ad = f_ground_truth(random_inargs);
            auto res_gen = casadi_func(random_inargs);
            for (size_t i : range(sx_outputs.size())) {
                auto diff = (double)cs::DM::norm_inf(res_ad[i] - res_gen[i]);
                if (diff > check_res_inf[i])
                    check_res_inf[i] = diff;
            }
        }
        std::cout << func_name << " check inf residual:\n";
        for (size_t i = 0; i < sx_inputs.size(); ++i) {
            std::cout << "\t" << opt.sx_inputs[i]->name_ << ":\t" << check_res_inf[i] << "\n";
        }
    }

    // Step 2: Generate raw C code
    cs::CodeGenerator cgen("raw");
    cgen.add(casadi_func);
    auto output_dir = std::filesystem::path(opt.output_dir);
    std::filesystem::create_directory(output_dir);
    std::string cpp_path = output_dir / (func_name + "_");
    cgen.generate(cpp_path);

    // Step 3: Parse raw C code and replace array access with Eigen::Ref access
    std::ifstream fin(cpp_path + "raw.c");
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(fin, line))
        lines.push_back(line);
    fin.close();

    // Extract input/output variable shapes
    std::vector<std::pair<int, int>> input_shapes, output_shapes;
    for (const auto &x : sx_inputs)
        input_shapes.emplace_back(x.size1(), x.size2());
    for (const auto &y : outputs_vec)
        output_shapes.emplace_back(y.size1(), y.size2());

    // Make the mapping of ccs indices to (i, j)
    std::vector<std::vector<std::pair<int, int>>> ij_pairs_all;
    for (const auto &x : sx_inputs)
        ij_pairs_all.push_back(ccs_index_to_ij(x.size1(), x.size2(), x.get_row(), x.get_colind()));
    for (const auto &y : outputs_vec)
        ij_pairs_all.push_back(ccs_index_to_ij(y.size1(), y.size2(), y.get_row(), y.get_colind()));

    // Step 4: Replace all casadi generated arrays with Eigen::Ref notation
    std::vector<std::string> processed_lines;
    bool func_found = false, func_done = false;
    bool vec_out = outputs_vec[0].size2() == 1;
    bool mat_output = !vec_out;
    bool is_hessian = func_name.size() >= 5 && func_name.substr(func_name.size() - 5) == "_hess";

    std::vector<std::string> non_op_sets;

    auto detect_non_op_set = [&non_op_sets](const std::string &code_str) {
        static const boost::regex pattern(R"(\b(a\d+)\b\s*=\s*(?:0*\.?0*123456(?:0+)?|1\.23456(?:0+)?e-?0*1)\b)");
        boost::smatch match;
        if (boost::regex_search(code_str, match, pattern)) {
            non_op_sets.push_back(match[1]);
            return true;
        }
        return false;
    };

    auto make_input_ref_access = [&](int arg_idx, int index) -> std::string {
        auto [i, j] = ij_pairs_all[arg_idx][index];
        return "inputs[" + std::to_string(arg_idx) + "](" + std::to_string(i) + ")";
    };
    auto make_output_ref_access = [&](int arg_idx, int index) -> std::string {
        if (is_hessian) {
            int n_in = sx_inputs.size();
            int row = arg_idx / n_in;
            int col = arg_idx % n_in;
            auto [i, j] = ij_pairs_all[arg_idx + n_in][index];
            return "outputs[" + std::to_string(row) + "][" + std::to_string(col) + "](" + std::to_string(i) + "," + std::to_string(j) + ")+";
        } else {
            auto [i, j] = ij_pairs_all[arg_idx + sx_inputs.size()][index];
            if (mat_output)
                return "outputs[" + std::to_string(arg_idx) + "](" + std::to_string(i) + "," + std::to_string(j) + ")";
            else
                return "outputs(" + std::to_string(i) + ")";
        }
    };

    for (const auto &l : lines) {
        std::string curr = l;
        if (!func_found) {
            if (curr.find("static int casadi_f0") != std::string::npos) {
                processed_lines.push_back("CASADI_SYMBOL_EXPORT void " + func_name + "(\n");
                processed_lines.push_back("  std::vector<Eigen::Ref<Eigen::VectorXd>>& inputs,\n");
                if (mat_output) {
                    if (is_hessian)
                        processed_lines.push_back("  std::vector<std::vector<Eigen::Ref<Eigen::MatrixXd>>>& outputs) {\n");
                    else
                        processed_lines.push_back("  std::vector<Eigen::Ref<Eigen::MatrixXd>>& outputs) {\n");
                } else {
                    processed_lines.push_back("  Eigen::Ref<Eigen::VectorXd> outputs) {\n");
                }
                func_found = true;
            }
            continue;
        }
        if (func_found && !func_done && curr.find("return 0;") != std::string::npos)
            continue;
        if (func_found && !func_done && curr.find("}") != std::string::npos) {
            processed_lines.push_back("}\n");
            func_done = true;
            break;
        }
        // detect non op (empty) a{xxx}
        if (!non_op_sets.empty()) {
            bool found_non_op = false;
            for (auto &i : non_op_sets) {
                if (curr.find(i + ";") != std::string::npos) {
                    found_non_op = true;
                    break;
                }
            }
            if (found_non_op)
                continue;
        }
        if (detect_non_op_set(curr))
            continue;

        static const boost::regex conditional_re(R"(arg\[(\d+)\]\? ([^:;]+) : 0;)");

        curr = boost::regex_replace(curr, conditional_re, "\\2;");

        static const boost::regex if_re(R"(if\s*\(res\[\d+\]!=0\)\s*\s*(.+);)");
        curr = boost::regex_replace(curr, if_re, "\\1;");

        static const boost::regex input_re(R"(arg\[(\d+)\]\[(\d+)\])");
        static const boost::regex output_re(R"(res\[(\d+)\]\[(\d+)\])");

        curr = boost::regex_replace(
            curr, input_re,
            [&](const boost::smatch &m) { return make_input_ref_access(std::stoi(m[1]), std::stoi(m[2])); },
            boost::match_default | boost::format_all);

        curr = boost::regex_replace(
            curr, output_re,
            [&](const boost::smatch &m) { return make_output_ref_access(std::stoi(m[1]), std::stoi(m[2])); },
            boost::match_default | boost::format_all);

        processed_lines.push_back(curr);
    }

    // Step 5: Write new C++ file with Eigen interface
    std::string final_cpp_path = output_dir / (func_name + ".cpp");
    std::ofstream fout(final_cpp_path);
    fout << "#include <vector>\n#include <Eigen/Dense>\n\n";
    fout << "#define casadi_real double\n\n";
    fout << "#ifdef __cplusplus\n";
    fout << "#if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)\n";
    fout << "    #if defined(STATIC_LINKED)\n";
    fout << "    #define CASADI_SYMBOL_EXPORT\n";
    fout << "    #else\n";
    fout << "    #define CASADI_SYMBOL_EXPORT __declspec(dllexport)\n";
    fout << "    #endif\n";
    fout << "#elif defined(__GNUC__)\n";
    fout << "    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility (\"default\")))\n";
    fout << "#endif\n";
    fout << "extern \"C\" {\n";
    fout << "#endif\n\n";
    for (const auto &pl : processed_lines)
        fout << pl << "\n";
    fout << "\n#ifdef __cplusplus\n";
    fout << "}\n";
    fout << "#endif\n\n";
    fout.close();

    std::cout << "Generated: " << final_cpp_path << std::endl;
    if (opt.compile) {
        // Step 6: Compile the generated C++ code into a shared library
        std::string so_file_path = output_dir / ("lib" + func_name + ".so");
        std::string compile_command =
            "g++ -shared -fPIC -O3 -DNDEBUG -std=gnu++20 -march=native -o " + so_file_path + " " + final_cpp_path + " -I /usr/include/eigen3";
        std::system(compile_command.c_str());
        std::cout << "Compiled:  " << so_file_path << std::endl;
        if (!opt.keep_raw) {
            std::remove((cpp_path + "raw.c").c_str());
            std::cout << "Removed:   " << cpp_path + "raw.c" << std::endl;
        }
        if (!opt.keep_c_src) {
            std::remove(final_cpp_path.c_str());
            std::cout << "Removed:   " << final_cpp_path << std::endl;
        }
    }
}

static std::vector<std::future<void>> workers;

void generate_and_compile(const std::string &func_name,
                          codegen_opts opt,
                          cs::SX sx_output) {
    size_t n_in = opt.sx_inputs.size();
    if (opt.gen_eval)
        workers.push_back(std::async(std::launch::async, [func_name, opt, sx_output]() {
            generate_eigen_cpp(func_name, opt, {sx_output});
        }));
    std::set<size_t> excluded;
    for (const auto &e : opt.exclude) {
        excluded.insert(e->uid_);
    }
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    auto gen_jacobian = opt.gen_jacobian;
    if (gen_jacobian) {
        std::map<size_t, cs::SX> external_jac;
        for (const auto &[arg, jac] : opt.ext_jac) {
            external_jac[arg->uid_] = jac;
        }
        cs::SXVector jacs;
        for (const auto &s : opt.sx_inputs) {
            if (excluded.contains(s->uid_)) {
                jacs.push_back(cs::SX(0.123456));
            } else if (external_jac.contains(s->uid_)) {
                jacs.push_back(external_jac.at(s->uid_));
            } else {
                jacs.push_back(jacobian(sx_output, s));
            }
        }
        if (!jacs.empty()) {
            if (opt.check_jac_ad) {
                cs::SXVector ad_jacs;
                for (const auto &s : opt.sx_inputs) {
                    ad_jacs.push_back(jacobian(sx_output, s));
                }
                try {
                    cs::Function f_ad(func_name + "_ad", to_sx_list(opt.sx_inputs), ad_jacs);
                    // Here you can wrap f_ad if needed
                    workers.push_back(std::async(std::launch::async, [func_name, opt, jacs, f_ad]() {
                        generate_eigen_cpp(func_name + "_jac", opt, jacs, true, f_ad);
                    }));
                } catch (const std::exception& ex) {
                    throw ex;
                }
            } else {
                workers.push_back(std::async(std::launch::async, [func_name, opt, jacs]() {
                    generate_eigen_cpp(func_name + "_jac", opt, jacs);
                }));
            }
        }
    }
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    auto gen_hessian = opt.gen_hessian;
    if (gen_hessian) {
        std::map<std::pair<size_t, size_t>, cs::SX> external_hess;
        for (const auto &[a, b, h] : opt.ext_hess) {
            external_hess[{a->uid_, b->uid_}] = h;
        }

        cs::SXVector hess_flat;
        for (const auto &i : opt.sx_inputs) {
            for (const auto &j : opt.sx_inputs) {
                auto ni = i->uid_;
                auto nj = j->uid_;

                if (excluded.contains(ni) || excluded.contains(nj) || (i->field_ < j->field_)) {
                    hess_flat.push_back(cs::SX(0.123456));
                    continue;
                }

                auto it = external_hess.find({ni, nj});
                if (it != external_hess.end()) {
                    hess_flat.push_back(it->second);
                    continue;
                }

                it = external_hess.find({nj, ni});
                if (it != external_hess.end()) {
                    hess_flat.push_back(it->second);
                    continue;
                }

                hess_flat.push_back(jacobian(jacobian(sx_output, i), j));
            }
        }
        workers.push_back(std::async(std::launch::async, [func_name, opt, hess_flat]() {
            generate_eigen_cpp(func_name + "_hess", opt, hess_flat);
        }));
    }
}

void wait_until_generated() {
    for (auto &f : workers) {
        f.get();
    }
    std::cout << "codegen done\n";
}

} // namespace atri