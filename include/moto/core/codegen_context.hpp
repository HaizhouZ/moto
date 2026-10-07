#ifndef MOTO_CORE_CODEGEN_CONTEXT_HPP
#define MOTO_CORE_CODEGEN_CONTEXT_HPP

#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace moto {

/// Immutable ownership of generated artifacts. No process-wide current context.
class codegen_context {
    const std::filesystem::path output_dir_;
    const std::filesystem::path linear_dir_;

    static std::filesystem::path resolve(const std::filesystem::path &path) {
        if (path.empty())
            throw std::invalid_argument("codegen output directory must not be empty");
        return std::filesystem::weakly_canonical(std::filesystem::absolute(path));
    }

  public:
    explicit codegen_context(const std::filesystem::path &output_dir = "gen")
        : output_dir_(resolve(output_dir)), linear_dir_(resolve(output_dir_ / "linear_backend")) {
        std::filesystem::create_directories(linear_dir_);
    }
    const std::filesystem::path &output_dir() const { return output_dir_; }
    const std::filesystem::path &linear_dir() const { return linear_dir_; }

    void require_compatible(const codegen_context &other, std::string_view owner) const {
        if (output_dir_ != other.output_dir_)
            throw std::invalid_argument(
                std::string(owner) + " uses codegen directory '" + output_dir_.string() +
                "', but requested '" + other.output_dir_.string() +
                "'; create a fresh model for a different directory");
    }
};

using codegen_context_ptr = std::shared_ptr<const codegen_context>;

inline codegen_context_ptr resolve_codegen(codegen_context_ptr context = {}) {
    return context ? std::move(context) : std::make_shared<const codegen_context>();
}

} // namespace moto
#endif
