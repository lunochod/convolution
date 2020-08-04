#ifndef CONVOLUTION_CORE_MATRIXTRANSPOSE_H
#define CONVOLUTION_CORE_MATRIXTRANSPOSE_H

#include <convolution/core/MatrixMultiplication.h>
#include <convolution/core/logging.h>

#include <cstdint>

namespace convolution {
namespace core {

/// simple out-of-place matrix transpose for matrices M != N
/// faster algorithms exist
template <typename T, MatrixOrder order>
void transpose(uint32_t M, uint32_t N, T *data, T *buffer) {
  if constexpr (order == core::MatrixOrder::kRowMajor) {
    for (uint32_t m = 0; m < M; ++m) {
      for (uint32_t n = 0; n < N; ++n) {
        buffer[n * M + m] = data[m * N + n];
      }
    }
  } else {
    for (uint32_t n = 0; n < N; ++n) {
      for (uint32_t m = 0; m < M; ++m) {
        buffer[m * N + n] = data[n * M + m];
      }
    }
  }
  memcpy(data, buffer, sizeof(T) * M * N);
}

}  // namespace core
}  // namespace convolution

#endif  // CONVOLUTION_CORE_MATRIXTRANSPOSE_H