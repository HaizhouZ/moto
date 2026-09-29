#pragma once

#include <filesystem>
#include <string>
#include <string_view>

namespace moto::utils {

class process_file_lock {
  public:
    explicit process_file_lock(const std::filesystem::path &lock_path);
    process_file_lock(const process_file_lock &) = delete;
    process_file_lock &operator=(const process_file_lock &) = delete;
    ~process_file_lock();

  private:
    int fd_ = -1;
};

struct runtime_compile_toolchain {
    std::string cxx;
    std::filesystem::path eigen_include;

    std::string fingerprint() const;
};

/// Resolve the compiler and Eigen headers used by Moto's runtime-generated code.
runtime_compile_toolchain runtime_compile_toolchain_config();

/// Quote one shell argument for the POSIX runtime compilation command.
std::string shell_quote(std::string_view value);

} // namespace moto::utils
