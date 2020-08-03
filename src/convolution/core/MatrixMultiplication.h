#ifndef CONVOLUTION_CORE_MATRIXMULTIPLICATION_H
#define CONVOLUTION_CORE_MATRIXMULTIPLICATION_H

#include <cstdint>

namespace convolution {
namespace core {

enum class MatrixOrder {
  kRowMajor,
  kColumnMajor
};

template <typename T, MatrixOrder order, bool useOverflowDetection = false>
bool gemm(uint32_t M, uint32_t N, uint32_t K, T *c, const T *a, const T *b) {
  static_assert(order == MatrixOrder::kRowMajor || order == MatrixOrder::kColumnMajor, "MatrixOrder provided is not supported.");

  T a_mk = 0;
  T b_kn = 0;

  for (uint32_t m = 0; m < M; ++m) {
    for (uint32_t n = 0; n < N; ++n) {
      T sum = (T)0;
      for (uint32_t k = 0; k < K; ++k) {
        if constexpr (order == MatrixOrder::kRowMajor) {
          a_mk = a[m * K + k];
          b_kn = b[k * N + n];
        } else {
          a_mk = a[k * M + m];
          b_kn = b[n * M + k];
        }
        if constexpr (useOverflowDetection) {
          T overflow = 0;
          if (__builtin_mul_overflow(a_mk, b_kn, &overflow)) {
            return false;
          }
          if (__builtin_add_overflow(sum, a_mk * b_kn, &overflow)) {
            return false;
          }
        }

        sum += a_mk * b_kn;
      }
      if constexpr (order == MatrixOrder::kRowMajor) {
        c[m * N + n] = sum;
      } else {
        c[n * M + m] = sum;
      }
    }
  }

  return true;
}

template <MatrixOrder order>
bool mult(uint32_t M, uint32_t N, uint8_t *c, const uint8_t *a, const uint8_t *b) {
  return gemm<uint8_t, order>(M, N, N, c, a, b);
}

}  // namespace core
}  // namespace convolution

#endif  // CONVOLUTION_CORE_MATRIXMULTIPLICATION_H