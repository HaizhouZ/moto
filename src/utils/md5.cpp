#include <filesystem>
#include <fstream>
#include <openssl/evp.h>

namespace moto {
namespace utils {
std::string compute_md5(const std::string &file_path) {
    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned int digest_len = 0;

    std::ifstream file(file_path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Could not open file for MD5 computation: " + file_path);
    }

    EVP_MD_CTX *ctx = EVP_MD_CTX_new();
    if (!ctx) {
        throw std::runtime_error("Failed to create EVP_MD_CTX");
    }

    if (EVP_DigestInit_ex(ctx, EVP_md5(), nullptr) != 1) {
        EVP_MD_CTX_free(ctx);
        throw std::runtime_error("Failed to initialize MD5 digest");
    }

    char buffer[8192];
    while (file.read(buffer, sizeof(buffer)) || file.gcount()) {
        if (EVP_DigestUpdate(ctx, buffer, file.gcount()) != 1) {
            EVP_MD_CTX_free(ctx);
            throw std::runtime_error("Failed to update MD5 digest");
        }
    }

    if (EVP_DigestFinal_ex(ctx, digest, &digest_len) != 1) {
        EVP_MD_CTX_free(ctx);
        throw std::runtime_error("Failed to finalize MD5 digest");
    }

    EVP_MD_CTX_free(ctx);

    std::string md5_str;
    md5_str.reserve(digest_len * 2);
    char hex_buffer[3];
    for (unsigned int i = 0; i < digest_len; ++i) {
        snprintf(hex_buffer, sizeof(hex_buffer), "%02x", digest[i]);
        md5_str.append(hex_buffer);
    }
    return md5_str;
}
} // namespace utils
} // namespace moto