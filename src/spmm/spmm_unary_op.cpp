
#include <moto/spmm/sparse_mat.hpp>

#include <moto/spmm/impl/spmm_impl.hpp>

#include <moto/spmm/impl/buffer.hpp>

namespace moto {
void sparse_mat::inner_product(const matrix &m, matrix &out) {
    using buffer = spmm::buffer;
    thread_local buffer cache_rhs;
    assert(m.rows() == m.cols() && m.cols() == rows_ && "matrix size mismatch");
    cache_rhs.resize(rows_, cols_);
    cache_rhs.data_.setZero();
    right_times(m, cache_rhs.data_);
    T_times(cache_rhs.data_, out);
}

} // namespace moto
