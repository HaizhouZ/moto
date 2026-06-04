#include <moto/spmm/impl/spmm_binary_op.hpp>

#include <moto/spmm/impl/binary_op_helper.hpp>

namespace moto {

template <bool add, typename rhs_type, typename out_type>
void sparse_mat::times(const rhs_type &rhs, out_type &out, clip_info info) const {
    if constexpr (std::is_same_v<rhs_type, vector> && std::is_same_v<out_type, row_vector>) {
        vector::AlignedMapType out_map(out.data(), out.cols());
        spmm::product<false, false, add>(*this, rhs, out_map, info);
    } else {
        assert((spmm::is_consistent<rhs_type, out_type>::value(spmm::times)) && "inconsistent dimensions for times");
        spmm::product<false, false, add>(*this, rhs, out, info);
    }
}
EXPLICIT_SP_MEMFUNC_INSTANTIATE(times)

} // namespace moto
