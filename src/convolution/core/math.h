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

/// \brief address calculation for MxN matrices
template <MatrixOrder order>
uint64_t address(uint32_t M, uint32_t N, uint32_t m, uint32_t n) {
  if constexpr (order == core::MatrixOrder::kRowMajor) {
    return N * m + n;
  } else {
    return M * n + m;
  }
}

/// \brief simple out-of-place matrix transpose for general matrices where M != N
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

/// \brief general MxNxK matrix-matrix multiplication c = a * b
///  Matrix c has dimensions MxN
///  Matrix a has dimensions MxK
///  Matrix b has dimensions KxN
/// \tparam R(typename) type used by the matrix c storing the result of a * b
/// \tparam T(typename) type used bu the input matrix a and b
/// \tparam cOrder(MatrixOrder) the storage format used by matrix c
/// \tparam aOrder(MatrixOrder) the storage format used by matrix a
/// \tparam bOrder(MatrixOrder) the storage format used by matrix b
/// \tparam useOverflowDetection(bool) flag used to enable / disable overflow detection at compile time, for instance when R == T
/// \param M(uint32_t) matrix dimension
/// \param N(uint32_t) matrix dimension
/// \param K(uint32_t) matrix dimension
/// \param c(R *) raw pointer to output data representing matrix c
/// \param a(const T *) raw pointer to input data representing matrix a
/// \param b(const T *) raw pointer to input data representing matrix b
/// \return bool true on success, false otherwise
template <typename R, typename T, MatrixOrder cOrder, MatrixOrder aOrder, MatrixOrder bOrder, bool useOverflowDetection = false>
bool gemm(uint32_t M, uint32_t N, uint32_t K, R *c, const T *a, const T *b) {
  R a_mk = 0;
  R b_kn = 0;

  for (uint32_t m = 0; m < M; ++m) {
    for (uint32_t n = 0; n < N; ++n) {
      R sum = 0;
      for (uint32_t k = 0; k < K; ++k) {
        a_mk = a[address<aOrder>(M, K, m, k)];
        b_kn = b[address<bOrder>(K, N, k, n)];

        if constexpr (useOverflowDetection) {
          R overflow = 0;
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
        R overflow = 0;
        if (__builtin_add_overflow(c[address<cOrder>(M, N, m, n)], sum, &overflow)) {
          return false;
        }
      }
      c[address<cOrder>(M, N, m, n)] += sum;
    }
  }

  return true;
}

/// \brief general MxNxK matrix-matrix multiplication c = a * b
template <typename R, typename T, MatrixOrder order, bool useOverflowDetection = false>
bool gemm(uint32_t M, uint32_t N, uint32_t K, R *c, const T *a, const T *b) {
  return gemm<R, T, order, order, order, useOverflowDetection>(M, N, K, c, a, b);
}

/// \brief general MxNxK matrix-matrix multiplication using an MxPxP matrix-matrix multiplier
/// Note:
///  The storage format for the matrices are constrained to allow efficient traversal and selection
///  of ranges in the input and output matrices.
///
/// \tparam R(typename) type used by the matrix c storing the result of a * b
/// \tparam T(typename) type used bu the input matrix a and b
/// \tparam cOrder(MatrixOrder) the storage format used by matrix c, must be core::MatrixOrder::kColumnMajor
/// \tparam aOrder(MatrixOrder) the storage format used by matrix a, must be core::MatrixOrder::kColumnMajor
/// \tparam bOrder(MatrixOrder) the storage format used by matrix b, must be core::MatrixOrder::kRowMajor
/// \tparam P(uint32_t) size P of the MxPxP matrix multiplier
/// \tparam useOverflowDetection(bool) flag used to enable / disable overflow detection at compile time, for instance when R == T
/// \param M(uint32_t) matrix dimension
/// \param N(uint32_t) matrix dimension, must be divisible by P
/// \param K(uint32_t) matrix dimension, must be divisible by P
/// \param c(R *) raw pointer to output data representing matrix c
/// \param a(const T *) raw pointer to input data representing matrix a
/// \param b(const T *) raw pointer to input data representing matrix b
/// \return bool true on success, false otherwise
template <typename R, typename T, MatrixOrder cOrder, MatrixOrder aOrder, MatrixOrder bOrder, uint32_t P, bool useOverflowDetection = false>
bool mult(uint32_t M, uint32_t N, uint32_t K, R *c, const T *a, const T *b) {
  static_assert(aOrder == core::MatrixOrder::kColumnMajor, "Matrix a in c = a x b must be in core::MatrixOrder::kColumnMajor");
  static_assert(bOrder == core::MatrixOrder::kRowMajor, "Matrix b in c = a x b must be in core::MatrixOrder::kRowMajor");
  static_assert(cOrder == core::MatrixOrder::kColumnMajor, "Matrix c in c = a x b must be in core::MatrixOrder::kColumnMajor");

  if (N % P + K % P != 0) {
    spdlog::error("matrix dimensions MxNxK = {}x{}x{} not aligned with P = {}", M, N, K, P);
    return false;
  }

  bool noOverflow = true;

  const T *aPtr = a;
  const T *bPtr = b;
  R *cPtr = c;

  // we require one buffer to transpose N*P elements of matrix b
  std::vector<T> buffer(N * P);
  const T *bufferPtr = buffer.data();

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
      // the MxPxP matrix-matrix multiplication
      noOverflow &= gemm<R, T, cOrder, aOrder, core::MatrixOrder::kColumnMajor, useOverflowDetection>(M, P, P, cPtr, aPtr, bufferPtr);
      bufferPtr += P * P;  // step forward P*P elements in buffer
      cPtr += M * P;       // step forward M*P elements in matrix c
    }

    aPtr += M * P;  // step forward M*P elements in matrix a
    bPtr += N * P;  // step forward N*P elements in matrix b
  }

  return noOverflow;
}

/// \brief returns size aligned to alignment
template <typename T, T alignment>
constexpr T getAlignedSize(const T size) {
  return size % alignment == 0 ? size : (size / alignment + 1) * alignment;
}

}  // namespace core
}  // namespace convolution

#endif  // CONVOLUTION_CORE_MATH_H