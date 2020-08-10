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

/// simple out-of-place matrix transpose for matrices M != N
/// faster algorithms exist
template <typename T, MatrixOrder order>
void transpose(uint32_t M, uint32_t N, T *data, T *buffer = nullptr) {
  std::vector<T> tmp;
  if (!buffer) {
    tmp.resize(M * N);
    buffer = tmp.data();
  }
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
  static_assert(cOrder == core::MatrixOrder::kColumnMajor, "Matrix c in c = a x b must be in core::MatrixOrder::kColumnMajor");

  if (N % P + K % P != 0) {
    spdlog::error("matrix dimensions MxNxK = {}x{}x{} not aligned with P = {}", M, N, K, P);
    return false;
  }

  if (K > P) {
    // for K > P we break down the MxNxK multiplication into MxNxP multiplications
    bool noOverflow = true;

    const T *aPtr = a;  //< pointer into matrix a
    const T *bPtr = b;  //< pointer into matrix b
    T *cPtr = c;        //< pointer into matrix c

    std::vector<T> buffer(N * P);        //< a buffer of size N*P
    const T *bufferPtr = buffer.data();  // pointer into the buffer

    // outer loop over K in steps of P
    for (uint32_t p = 0; p < K; p += P) {
      // copy N*P elements from matrix b following bPtr into the buffer
      memcpy(buffer.data(), bPtr, buffer.size() * sizeof(T));

      // transpose data in buffer
      core::transpose<T, bOrder>(P, N, buffer.data());

      // reset pointers for inner loop
      bufferPtr = buffer.data();
      cPtr = c;

      // inner loop over N in steps of P
      for (uint32_t q = 0; q < N; q += P) {
        noOverflow &= gemm<T, cOrder, aOrder, core::MatrixOrder::kColumnMajor, useOverflowDetection>(M, P, P, cPtr, aPtr, bufferPtr);
        bufferPtr += P * P;  //< step forward P*P elements in buffer
        cPtr += M * P;       //< step forward M*P elements in matrix c
      }

      aPtr += M * P;  //< step forward M*P elements in matrix a
      bPtr += N * P;  //< step forward N*P elements in matrix b
    }

    return noOverflow;
  } else {
    // this multiplication is only called for K <= P representing the hardware multiplier
    return gemm<T, cOrder, aOrder, bOrder, useOverflowDetection>(M, N, K, c, a, b);
  }
}

template <typename T, T alignment>
constexpr T getAlignedSize(const T size) {
  return size % alignment == 0 ? size : (size / alignment + 1) * alignment;
}

}  // namespace core
}  // namespace convolution

#endif  // CONVOLUTION_CORE_MATH_H