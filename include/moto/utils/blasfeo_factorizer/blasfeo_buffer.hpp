#ifndef MOTO_UTILS_BLASFEO_FACTORIZATION_HPP
#define MOTO_UTILS_BLASFEO_FACTORIZATION_HPP

#include <blasfeo.h>
#include <moto/core/fwd.hpp>

namespace moto {
namespace utils {
template <typename T>
struct buffer_tpl {
    T::AlignedMapType data_;
    using PlainObjectType = T;
    using data_type = T::AlignedMapType;
    constexpr static size_t alignment = Eigen::internal::traits<decltype(data_)>::Alignment;
    void *mem_;
    size_t size_;

    buffer_tpl() : data_(nullptr, 0, 0), mem_(nullptr), size_(0) {}
    void resize(size_t r, size_t c) {
        size_t required_size = r * c;
        if (size_ < required_size) {
            if (mem_ != nullptr)
                ::operator delete(mem_, std::align_val_t(alignment));
            mem_ = ::operator new(sizeof(scalar_t) * required_size, std::align_val_t(alignment));
            size_ = required_size;
        }

        if (r != data_.rows() || c != data_.cols())
            new (&data_) data_type(reinterpret_cast<scalar_t *>(mem_), r, c);
    }
    ~buffer_tpl() {
        if (mem_ != nullptr)
            ::operator delete(mem_, std::align_val_t(alignment));
    }
};
template <typename DataType>
struct blasfeo_buffer_tpl {
    DataType data_;
    void *mem_ = nullptr;
    size_t size_ = 0;

    blasfeo_buffer_tpl() : mem_(nullptr), size_(0) {}
    void resize(size_t r, size_t c) {
        size_t required_size;
        if constexpr (std::is_same_v<DataType, blasfeo_dmat>)
            required_size = blasfeo_memsize_dmat(r, c);
        else {
            assert(c == 1); // blasfeo_dvec only has one column
            required_size = blasfeo_memsize_dvec(r);
        }
        if (size_ < required_size) {
            if (mem_ != nullptr)
                v_free_align(mem_);
            v_zeros_align(&mem_, required_size);
            size_ = required_size;
        }
        if constexpr (std::is_same_v<DataType, blasfeo_dmat>) {
            blasfeo_create_dmat(r, c, &data_, mem_);
        } else {
            blasfeo_create_dvec(r, &data_, mem_);
        }
    }
    ~blasfeo_buffer_tpl() {
        if (mem_ != nullptr)
            v_free_align(mem_);
    }
    // Helper function to convert Eigen::MatrixXd to blasfeo_dmat.
    // This allocates and packs the data into a BLASFEO matrix structure.
    template <typename T>
    void from_eigen(const T &eigen_mat) {
        // Pack the data from the Eigen matrix into the BLASFEO matrix
        resize(eigen_mat.rows(), eigen_mat.cols());
        if constexpr (std::is_same_v<DataType, blasfeo_dmat>)
            blasfeo_pack_dmat(eigen_mat.rows(), eigen_mat.cols(), (double *)eigen_mat.data(), eigen_mat.rows(), &data_, 0, 0);
        else
            blasfeo_pack_dvec(eigen_mat.rows(), (double *)eigen_mat.data(), 1, &data_, 0);
    }
    // Helper function to convert blasfeo_dmat to Eigen::MatrixXd.
    // This unpacks data from a BLASFEO matrix into an Eigen matrix.
    template <typename T>
    void to_eigen(T &eigen_mat) {
        // Resize the Eigen matrix to match BLASFEO matrix dimensions
        // Unpack the data from the BLASFEO matrix into the Eigen matrix
        if constexpr (std::is_same_v<DataType, blasfeo_dmat>) {
            eigen_mat.resize(data_.m, data_.n);
            blasfeo_unpack_dmat(data_.m, data_.n, &data_, 0, 0, (double *)eigen_mat.data(), eigen_mat.rows());
        } else {
            eigen_mat.resize(data_.m);
            blasfeo_unpack_dvec(data_.m, &data_, 0, (double *)eigen_mat.data(), 1);
        }
    }
};

using blasfeo_buffer = blasfeo_buffer_tpl<blasfeo_dmat>;
using blasfeo_vec_buffer = blasfeo_buffer_tpl<blasfeo_dvec>;

} // namespace utils
} // namespace moto

#endif
