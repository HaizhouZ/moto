#include <catch2/catch_test_macros.hpp>
#include <moto/core/expr.hpp>
#include <moto/core/linear_backend.hpp>
#include <casadi/casadi.hpp>

#include <chrono>
#include <map>
#include <unistd.h>

namespace moto {
namespace {
struct temporary_directory {
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("moto_context_" + std::to_string(getpid()) + "_" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    ~temporary_directory() { std::filesystem::remove_all(path); }
};

auto libraries(const std::filesystem::path &root) {
    std::map<std::filesystem::path, std::filesystem::file_time_type> result;
    for (const auto &entry : std::filesystem::recursive_directory_iterator(root))
        if (entry.path().extension() == ".so")
            result.emplace(entry.path().filename(), entry.last_write_time());
    return result;
}

struct tracked_expr : expr {
    size_t finalizations = 0;
    mutable size_t waits = 0;

    explicit tracked_expr(const std::string &name) : expr(name, 1, __usr_var) {}

    void finalize_impl() override {
        for (const auto &dependency : dep()) {
            REQUIRE(dependency->finalized());
            REQUIRE(dependency->codegen() == codegen());
        }
        ++finalizations;
        expr::finalize_impl();
    }

    bool wait_until_ready() const override {
        ++waits;
        return expr::wait_until_ready();
    }
};
} // namespace

TEST_CASE("shared expression dependencies are finalized and waited once per call") {
    temporary_directory temp;
    auto context = std::make_shared<codegen_context>(temp.path);
    std::vector<std::shared_ptr<tracked_expr>> graph;
    // Each level shares both of the preceding level's nodes. A recursive walk
    // without deduplication would revisit the leaves exponentially often.
    for (size_t i = 0; i < 24; ++i) {
        auto node = std::make_shared<tracked_expr>("dependency_" + std::to_string(i));
        if (i >= 2) {
            const size_t previous = (i / 2 - 1) * 2;
            node->add_dep(*graph[previous]);
            node->add_dep(*graph[previous + 1]);
        }
        graph.push_back(std::move(node));
    }
    auto root = std::make_shared<tracked_expr>("root");
    root->add_dep(*graph[22]);
    root->add_dep(*graph[23]);
    graph.push_back(root);
    REQUIRE(root->finalize(false, context));
    for (const auto &node : graph) {
        REQUIRE(node->finalizations == 1);
        REQUIRE(node->waits == 0);
    }
    for (size_t call = 1; call <= 2; ++call) {
        REQUIRE(root->finalize(true, context));
        for (const auto &node : graph) {
            REQUIRE(node->finalizations == 1);
            REQUIRE(node->waits == call);
        }
    }
}

TEST_CASE("binding revalidates dependencies added after earlier finalization") {
    temporary_directory temp;
    auto a = std::make_shared<codegen_context>(temp.path / "a");
    auto b = std::make_shared<codegen_context>(temp.path / "b");
    auto root = std::make_shared<tracked_expr>("root");
    REQUIRE(root->finalize(true, a));
    auto added = std::make_shared<tracked_expr>("added");
    auto conflicting = std::make_shared<tracked_expr>("conflicting");
    conflicting->bind_codegen(b);
    added->add_dep(*conflicting);
    root->add_dep(*added);
    REQUIRE_THROWS_AS(root->finalize(true, a), std::invalid_argument);
    REQUIRE_FALSE(added->codegen());
    REQUIRE(added->finalizations == 0);
    REQUIRE(conflicting->finalizations == 0);

    added->dep().clear();
    REQUIRE(root->finalize(true, a));
    REQUIRE(root->finalizations == 1);
    REQUIRE(added->finalizations == 1);
    REQUIRE(added->waits == 1);
    REQUIRE(added->codegen() == a);
}

TEST_CASE("sparse matrix copies preserve resolved directory ownership") {
    temporary_directory temp;
    const auto artifacts = temp.path / "artifacts";
    std::filesystem::create_directories(artifacts);
    std::filesystem::create_directories(temp.path / "root");
    std::filesystem::create_directory_symlink(artifacts, temp.path / "root" / "linear_backend");
    auto context = std::make_shared<codegen_context>(temp.path / "root");
    REQUIRE(context->linear_dir() == artifacts);

    sparse_matrix source;
    source.bind_codegen(context);
    source.resize(2, 2);
    source.insert(0, 0, 2, 2, sparsity::diag).setConstant(2.);
    sparse_matrix copy(source), snapshot;
    snapshot.bind_codegen(context);
    snapshot = source;
    REQUIRE(copy.linear_codegen_dir() == artifacts);
    REQUIRE(snapshot.linear_codegen_dir() == artifacts);
    REQUIRE(snapshot.dense().isApprox(2. * matrix::Identity(2, 2)));

    // Filesystem changes after binding must not reinterpret ownership during
    // an ordinary runtime value copy or another bind from the same context.
    std::filesystem::rename(artifacts, temp.path / "moved");
    std::filesystem::create_directory_symlink(temp.path / "moved", artifacts);
    REQUIRE_NOTHROW(snapshot = source);
    REQUIRE_NOTHROW(snapshot.bind_codegen(context));
    sparse_matrix later(source);
    REQUIRE(later.linear_codegen_dir() == artifacts);

    auto other = std::make_shared<codegen_context>(temp.path / "other");
    sparse_matrix foreign;
    foreign.bind_codegen(other);
    REQUIRE_THROWS_AS(snapshot = foreign, std::invalid_argument);
    REQUIRE(snapshot.linear_codegen_dir() == artifacts);
    REQUIRE(snapshot.dense().isApprox(2. * matrix::Identity(2, 2)));
}

TEST_CASE("lazy linear kernels and their caches respect context directories") {
    using namespace linear_backend;
    temporary_directory temp;
    auto a = std::make_shared<codegen_context>(temp.path / "a space");
    auto b = std::make_shared<codegen_context>(temp.path / "b");
    const auto exercise = [](const codegen_context_ptr &context) {
        sparse_matrix sparse;
        sparse.bind_codegen(context);
        sparse.resize(2, 2);
        sparse.insert(0, 0, 2, 2, sparsity::diag).setConstant(2.);
        sparse_matrix copy(sparse);
        REQUIRE(copy.linear_codegen_dir() == context->linear_dir());
        matrix rhs = matrix::Ones(2, 1), out = matrix::Zero(2, 1);
        multiply(copy, rhs, out);
        REQUIRE(out.isApprox(2. * rhs));
        matrix square = matrix::Zero(2, 2);
        multiply(sparse, copy, square);
        REQUIRE(square.isApprox(4. * matrix::Identity(2, 2)));
        write_dense(copy, square, {.overwrite = true});
        REQUIRE(square.isApprox(2. * matrix::Identity(2, 2)));

        const std::array requests{product_request{
            &sparse, product_op::transpose_times, 1., 2, 1, 2, 1}};
        prepare_products(requests);
        out.setZero();
        transpose_multiply(sparse, rhs, out);
        REQUIRE(out.isApprox(2. * rhs));

        const product_spec spec{describe(sparse), product_op::times, 2, 1, 2, 1};
        auto batch = compile_batch_product({{spec}}, context->linear_dir());
        auto pointers = panel_pointers(sparse);
        pointers.push_back(rhs.data());
        pointers.push_back(out.data());
        out.setZero();
        batch(pointers);
        REQUIRE(out.isApprox(2. * rhs));

        compile_batch_jacobian_product({{2, describe(sparse).panels}}, context->linear_dir());
        condensation_spec condensation;
        condensation.rows = 2;
        condensation.jacobians = describe(sparse).panels;
        condensation.residual_signs = {1.};
        compile_batch_condensation({{condensation}}, context->linear_dir());
        compile_rowwise(describe(sparse), context->linear_dir());
    };
    exercise(a);
    const auto first = libraries(a->linear_dir());
    REQUIRE(first.size() >= 8);
    exercise(b);
    const auto second = libraries(b->linear_dir());
    REQUIRE(first.size() == second.size());
    for (const auto &[file, _] : first) REQUIRE(second.contains(file));
    exercise(std::make_shared<codegen_context>(temp.path / "a space" / "."));
    REQUIRE(libraries(a->linear_dir()) == first);

    sparse_matrix ma, mb;
    ma.bind_codegen(a);
    mb.bind_codegen(b);
    const std::array mixed{product_request{&ma}, product_request{&mb}};
    REQUIRE_THROWS_AS(prepare_products(mixed), std::invalid_argument);
}

TEST_CASE("live graph plans are cached separately for each output directory") {
    using namespace linear_backend;
    temporary_directory temp;
    auto a = std::make_shared<codegen_context>(temp.path / "a");
    auto b = std::make_shared<codegen_context>(temp.path / "b");
    auto x = casadi::MX::sym("x", 2, 1);
    auto scale = casadi::MX::sym("scale", 2, 2);
    std::vector<sparse_matrix> workspace_a, workspace_b;
    auto ga = compile_graph({scale, x}, {casadi::MX::mtimes(scale, x)}, &workspace_a, a->linear_dir());
    auto gb = compile_graph({scale, x}, {casadi::MX::mtimes(scale, x)}, &workspace_b, b->linear_dir());
    auto ga_copy = ga.instantiate();
    matrix input = matrix::Ones(2, 1), output = matrix::Zero(2, 1);
    matrix multiplier = 2. * matrix::Identity(2, 2);
    std::array pointers{multiplier.data(), input.data(), output.data()};
    for (auto *graph : {&ga, &gb, &ga_copy}) {
        (*graph)(pointers);
        REQUIRE(output.isApprox(2. * input));
    }
    REQUIRE(!libraries(a->linear_dir()).empty());
    REQUIRE(libraries(a->linear_dir()).size() == libraries(b->linear_dir()).size());
    for (const auto &matrix : workspace_b)
        REQUIRE(matrix.linear_codegen_dir() == b->linear_dir());
}
} // namespace moto
