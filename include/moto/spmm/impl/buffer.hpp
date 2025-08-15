#ifndef MOTO_SPMM_IMPL_BUFFER_HPP
#define MOTO_SPMM_IMPL_BUFFER_HPP

#include <moto/spmm/panel_mat.hpp>

namespace moto {
namespace spmm {

struct buffer : public panel_mat<sparsity::dense> {
    matrix::AlignedMapType data_;
    struct {
    } null_;
    void *mem_;
    size_t size_;
    using base = panel_mat<sparsity::dense>;
    using base::sparsity_type;

    buffer() : data_(nullptr, 0, 0), mem_(nullptr), panel_mat<sparsity::dense>(0, 0, 0, 0, false) {}
    template <typename T>
        requires(std::is_base_of_v<panel_mat_base, std::remove_cvref_t<T>>)
    void resize(T &p) {
        copy_dim(p);
        resize(p.rows_, p.cols_);
    }
    void resize(size_t r, size_t c) {
        size_t required_size = r * c;
        if (size_ < required_size) {
            if (mem_ != nullptr)
                ::operator delete(mem_, std::align_val_t(64));
            mem_ = ::operator new(sizeof(scalar_t) * required_size, std::align_val_t(64));
            size_ = required_size;
        }
        if (r != data_.rows() || c != data_.cols())
            new (&data_) matrix::AlignedMapType(reinterpret_cast<scalar_t *>(mem_), r, c);
    }
    ~buffer() {
        if (mem_ != nullptr)
            ::operator delete(mem_, std::align_val_t(64));
    }
};

} // namespace spmm
} // namespace moto

#endif