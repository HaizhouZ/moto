#ifndef MOTO_UTILS_CODEGEN_HPP
#define MOTO_UTILS_CODEGEN_HPP

#include <moto/core/fields.hpp> // Assuming moto::field_t is defined here
#include <moto/ocp/sym.hpp>

#include <future>
#include <mutex>

// 3rd-party includes
#include "nlohmann/json.hpp" // nlohmann/json
#include <casadi/casadi.hpp>
#include <re2/re2.h>

namespace moto {
namespace utils {

// Namespace aliases
namespace fs = std::filesystem;
using json = nlohmann::json;

std::string compute_md5(const std::string &file_path);

namespace cs_codegen {
struct worker {
    using future_type = std::shared_future<void>;
    future_type future;
    void wait_until_finished() {
        if (future.valid()) {
            future.get();
        }
    }
    worker(future_type &&f) : future(std::move(f)) {}
};

struct worker_list {
    std::vector<worker> workers;
    void wait_until_finished() {
        for (auto &w : workers) {
            w.wait_until_finished();
        }
    }
    void add(worker::future_type &&w) {
        workers.emplace_back(std::move(w));
    }
    void add(const worker_list &other) {
        workers.insert(workers.end(), other.workers.begin(), other.workers.end());
    }
};

struct task {
    std::string func_name;
    using in_arg_list_t = sym_list;
    in_arg_list_t sx_inputs;
    cs::SX sx_output;
    bool gen_eval = true;
    bool gen_jacobian = false;
    bool gen_hessian = false;
    std::vector<std::pair<expr_ref, cs::SX>> ext_jac;
    std::vector<std::tuple<expr_ref, expr_ref, cs::SX>> ext_hess;
    std::string output_dir = "gen";
    bool force_recompile = false;
    bool check_jac_ad = false; // check if jacobian is correct by comparing with ad
    bool append_value = false;
    bool append_jac = false;
    bool keep_generated_src = false; // keep generated files
    std::string eval_compile_flag = "-O3 -DNDEBUG -march=native";
    std::string jac_compile_flag = "-O3 -DNDEBUG -march=native";
    std::string hess_compile_flag = "-O3 -DNDEBUG -march=native";
    std::string prefix = "";
    bool verbose = false; // verbose output

    void finalize(worker_list &workers);
};

// Public entry point to start code generation
worker_list generate_and_compile(task &&_task);

// // Waits for all compilation threads to finish
void wait_until_generated();
}; // namespace cs_codegen
} // namespace utils
} // namespace moto

#endif // MOTO_UTILS_CODEGEN_HPP