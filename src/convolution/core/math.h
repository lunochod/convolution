#ifndef CONVOLUTION_CORE_MATH_H
#define CONVOLUTION_CORE_MATH_H

#include <convolution/core/logging.h>

#include <cstdint>

namespace convolution {
namespace core {

enum class MatrixOrder {
  kRowMajor,
  kColumnMajor
};

template <MatrixOrder order>
uint64_t address(uint32_t M, uint32_t N, uint32_t m, uint32_t n) {
  if constexpr (order == core::MatrixOrder::kRowMajor) {
    return N * m + n;
  } else {
    return M * n + m;
  }
}

template <typename T, MatrixOrder cOrder, MatrixOrder aOrder, MatrixOrder bOrder, bool useOverflowDetection = false>
bool gemm(uint32_t M, uint32_t N, uint32_t K, T *c, const T *a, const T *b) {
  T a_mk = 0;
  T b_kn = 0;

  for (uint32_t m = 0; m < M; ++m) {
    for (uint32_t n = 0; n < N; ++n) {
      T sum = (T)0;
      for (uint32_t k = 0; k < K; ++k) {
        a_mk = a[address<aOrder>(M, K, m, k)];
        b_kn = b[address<bOrder>(K, N, k, n)];

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

      if constexpr (useOverflowDetection) {
        T overflow = 0;
        if (__builtin_add_overflow(c[address<cOrder>(M, N, m, n)], sum, &overflow)) {
          return false;
        }
      }
      c[address<cOrder>(M, N, m, n)] += sum;
    }
  }

  return true;
}

template <typename T, MatrixOrder order, bool useOverflowDetection = false>
bool gemm(uint32_t M, uint32_t N, uint32_t K, T *c, const T *a, const T *b) {
  return gemm<T, order, order, order, useOverflowDetection>(M, N, K, c, a, b);
}

template <typename T, MatrixOrder cOrder, MatrixOrder aOrder, MatrixOrder bOrder, uint32_t P, bool useOverflowDetection = false>
bool mult(uint32_t M, uint32_t N, uint32_t K, T *c, const T *a, const T *b) {
  static_assert(aOrder == core::MatrixOrder::kColumnMajor, "Matrix a in c = a x b must be in core::MatrixOrder::kColumnMajor");
  static_assert(bOrder == core::MatrixOrder::kRowMajor, "Matrix b in c = a x b must be in core::MatrixOrder::kRowMajor");

  if (K > P) {
    // for K > P we break down the MxNxK multiplication into MxNxP multiplications
    bool res = true;

    const T *a1 = a;
    const T *b1 = b;

    const uint32_t a_height = M;
    const uint32_t b_width = N;

    // the partial results get accumulated in the output matrix c
    // this may result in overflows
    for (uint32_t p = P; p <= K; p += P) {
      res &= mult<T, cOrder, aOrder, bOrder, P, useOverflowDetection>(M, N, P, c, a1, b1);
      a1 += a_height * P;
      b1 += b_width * P;
    }

    if (K % P != 0) {
      // if K doesn't align with P we execute one more MxNx(K%P) multiplication
      // we assume that the P-multiplier can handle K < P internally
      // the alternative is to pad the input data, such that K % P == 0
      res &= mult<T, cOrder, aOrder, bOrder, P, useOverflowDetection>(M, N, K % P, c, a1, b1);
    }

    return res;
  } else {
    // this multiplication is only called for K <= P representing the hardware multiplier
    return gemm<T, cOrder, aOrder, bOrder, useOverflowDetection>(M, N, K, c, a, b);
  }
}

/// simple out-of-place matrix transpose for matrices M != N
/// faster algorithms exist
template <typename T, MatrixOrder order>
void transpose(uint32_t M, uint32_t N, T *data, T *buffer) {
  for (uint32_t m = 0; m < M; ++m) {
    for (uint32_t n = 0; n < N; ++n) {
      if constexpr (order == core::MatrixOrder::kRowMajor) {
        buffer[address<core::MatrixOrder::kColumnMajor>(M, N, m, n)] = data[address<core::MatrixOrder::kRowMajor>(M, N, m, n)];
      } else {
        buffer[address<core::MatrixOrder::kRowMajor>(M, N, m, n)] = data[address<core::MatrixOrder::kColumnMajor>(M, N, m, n)];
      }
    }
  }
  memcpy(data, buffer, sizeof(T) * M * N);
}

}  // namespace core
}  // namespace convolution

#endif  // CONVOLUTION_CORE_MATH_H