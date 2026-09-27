#ifndef MOTO_UTILS_BLASFEO_QR_HPP
#define MOTO_UTILS_BLASFEO_QR_HPP

#include <moto/utils/blasfeo_factorizer/blasfeo_lq.hpp>

namespace moto::utils {

// Compatibility name retained for the former LQ implementation, which was
// historically misnamed "qr".  New code uses blasfeo_lq directly.
using blasfeo_qr = blasfeo_lq;

} // namespace moto::utils

#endif
