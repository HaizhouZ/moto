#include "runtime_compiler.hpp"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <optional>
#include <stdexcept>
#include <sys/file.h>
#include <unistd.h>
#include <vector>

#ifndef _WIN32
#include <dlfcn.h>
#endif

namespace moto::utils {
namespace {

std::optional<std::string> nonempty_environment(const char *name) {
    if (const char *value = std::getenv(name); value != nullptr && *value != '\0')
        return std::string(value);
    return std::nullopt;
}

std::optional<std::filesystem::path>
eigen_include_from(std::filesystem::path candidate) {
    if (std::filesystem::exists(candidate / "Eigen" / "Core"))
        return candidate;
    if (std::filesystem::exists(candidate / "eigen3" / "Eigen" / "Core"))
        return candidate / "eigen3";
    return std::nullopt;
}

#ifndef _WIN32
std::optional<std::filesystem::path> installed_prefix() {
    Dl_info info{};
    if (dladdr(reinterpret_cast<void *>(&runtime_compile_toolchain_config),
               &info) == 0 || info.dli_fname == nullptr)
        return std::nullopt;
    std::error_code error;
    const auto library = std::filesystem::weakly_canonical(info.dli_fname, error);
    if (error || library.parent_path().empty())
        return std::nullopt;
    return library.parent_path().parent_path();
}
#endif

} // namespace

process_file_lock::process_file_lock(const std::filesystem::path &lock_path) {
    fd_ = ::open(lock_path.c_str(), O_CREAT | O_RDWR, 0666);
    if (fd_ == -1)
        throw std::runtime_error("failed to open runtime compilation lock " +
                                 lock_path.string() + ": " +
                                 std::strerror(errno));
    while (::flock(fd_, LOCK_EX) == -1) {
        if (errno == EINTR)
            continue;
        const std::string message = std::strerror(errno);
        ::close(fd_);
        fd_ = -1;
        throw std::runtime_error("failed to lock runtime compilation file " +
                                 lock_path.string() + ": " + message);
    }
}

process_file_lock::~process_file_lock() {
    if (fd_ != -1) {
        ::flock(fd_, LOCK_UN);
        ::close(fd_);
    }
}

std::string runtime_compile_toolchain::fingerprint() const {
    return cxx + "\n" + eigen_include.string();
}

runtime_compile_toolchain runtime_compile_toolchain_config() {
    runtime_compile_toolchain result;
    result.cxx = nonempty_environment("MOTO_CXX_COMPILER")
                     .value_or(nonempty_environment("CXX").value_or("c++"));

    std::vector<std::filesystem::path> candidates;
    if (const auto explicit_path = nonempty_environment("MOTO_EIGEN_INCLUDE_DIR"))
        candidates.emplace_back(*explicit_path);
    if (const auto conda_prefix = nonempty_environment("CONDA_PREFIX"))
        candidates.emplace_back(std::filesystem::path(*conda_prefix) / "include");
#ifndef _WIN32
    if (const auto prefix = installed_prefix())
        candidates.emplace_back(*prefix / "include");
#endif
    candidates.emplace_back("/usr/include/eigen3");
    candidates.emplace_back("/usr/local/include/eigen3");

    for (const auto &candidate : candidates)
        if (const auto found = eigen_include_from(candidate)) {
            result.eigen_include = *found;
            return result;
        }

    throw std::runtime_error(
        "Moto runtime code generation could not find Eigen headers; install "
        "Eigen in the active environment or set MOTO_EIGEN_INCLUDE_DIR to "
        "the directory containing Eigen/Core");
}

std::string shell_quote(std::string_view value) {
    std::string output = "'";
    for (const char character : value)
        output += character == '\'' ? "'\\''" : std::string(1, character);
    return output + "'";
}

} // namespace moto::utils
