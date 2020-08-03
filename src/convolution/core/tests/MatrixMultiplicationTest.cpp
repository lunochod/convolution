#include <convolution/core/MatrixMultiplication.h>
#include <convolution/core/logging.h>
#include <gtest/gtest.h>

#include <limits>
#include <random>

using namespace convolution;

namespace {

template <class T>
class MatrixMultiplicationTestFixture : public testing::Test {
};

template <typename T>
void initRandomMatrix(const uint32_t M, const uint32_t N, T *mat) {
  std::random_device device;
  std::mt19937 generator(device());
  std::uniform_int_distribution<T> distribution(1, std::numeric_limits<T>::max());

  for (uint32_t m = 0; m < M; ++m) {
    for (uint32_t n = 0; n < N; ++n) {
      mat[m * M + n] = distribution(generator);
    }
  }
}

template <typename T>
void initConstantMatrix(const uint32_t M, const uint32_t N, T *mat, T value) {
  for (uint32_t m = 0; m < M; ++m) {
    for (uint32_t n = 0; n < N; ++n) {
      mat[m * M + n] = value;
    }
  }
}

template <typename T>
void initIdentityMatrix(const uint32_t M, const uint32_t N, T *mat) {
  memset(mat, 0, M * M * sizeof(T));
  for (uint32_t m = 0; m < M; ++m) {
    mat[m * M + m] = 1;
  }
}

}  // namespace

typedef ::testing::Types<uint8_t> Implementations;
TYPED_TEST_SUITE(MatrixMultiplicationTestFixture, Implementations);

TYPED_TEST(MatrixMultiplicationTestFixture, MulZero) {
  constexpr uint32_t M = 3;
  constexpr uint32_t N = 4;
  constexpr uint32_t K = 5;

  std::vector<TypeParam> a(M * K);
  std::vector<TypeParam> b(N * K);
  std::vector<TypeParam> c(M * N);

  {
    memset(a.data(), 0, a.size() * sizeof(TypeParam));
    memset(b.data(), 1, b.size() * sizeof(TypeParam));
    memset(c.data(), 2, c.size() * sizeof(TypeParam));

    bool didNotOverflow = core::gemm<TypeParam, core::MatrixOrder::kRowMajor>(M, N, K, c.data(), a.data(), b.data());
    ASSERT_TRUE(didNotOverflow);

    for (uint32_t m = 0; m < M; ++m) {
      for (uint32_t n = 0; n < N; ++n) {
        ASSERT_EQ(c[m * N + n], 0);
      }
    }
  }
  {
    memset(a.data(), 1, a.size() * sizeof(TypeParam));
    memset(b.data(), 0, b.size() * sizeof(TypeParam));
    memset(c.data(), 2, c.size() * sizeof(TypeParam));

    bool didNotOverflow = core::gemm<TypeParam, core::MatrixOrder::kRowMajor>(M, N, K, c.data(), a.data(), b.data());
    ASSERT_TRUE(didNotOverflow);

    for (uint32_t m = 0; m < M; ++m) {
      for (uint32_t n = 0; n < N; ++n) {
        ASSERT_EQ(c[m * N + n], 0);
      }
    }
  }
}

TYPED_TEST(MatrixMultiplicationTestFixture, MulIdentity) {
  constexpr uint32_t M = 5;

  std::vector<TypeParam> a(M * M);
  std::vector<TypeParam> b(M * M);
  std::vector<TypeParam> c(M * M);

  initRandomMatrix<TypeParam>(M, M, a.data());
  initIdentityMatrix<TypeParam>(M, M, b.data());

  {
    memset(c.data(), 0, c.size() * sizeof(TypeParam));
    bool didNotOverflow = core::gemm<TypeParam, core::MatrixOrder::kRowMajor>(M, M, M, c.data(), a.data(), b.data());
    ASSERT_TRUE(didNotOverflow);

    for (uint32_t m = 0; m < M; ++m) {
      for (uint32_t n = 0; n < M; ++n) {
        ASSERT_EQ(c[m * M + n], a[m * M + n]);
      }
    }
  }
  {
    memset(c.data(), 0, c.size() * sizeof(TypeParam));
    bool didNotOverflow = core::gemm<TypeParam, core::MatrixOrder::kRowMajor>(M, M, M, c.data(), b.data(), a.data());
    ASSERT_TRUE(didNotOverflow);

    for (uint32_t m = 0; m < M; ++m) {
      for (uint32_t n = 0; n < M; ++n) {
        ASSERT_EQ(c[m * M + n], a[m * M + n]);
      }
    }
  }
}

TYPED_TEST(MatrixMultiplicationTestFixture, DetectOverflow) {
  constexpr uint32_t M = 5;

  std::vector<TypeParam> a(M * M);
  std::vector<TypeParam> b(M * M);
  std::vector<TypeParam> c(M * M);

  initConstantMatrix<TypeParam>(M, M, a.data(), std::numeric_limits<TypeParam>::max());
  initConstantMatrix<TypeParam>(M, M, b.data(), std::numeric_limits<TypeParam>::max());
  memset(c.data(), 0, c.size() * sizeof(TypeParam));
  bool didNotOverflow = core::gemm<TypeParam, core::MatrixOrder::kRowMajor, true>(M, M, M, c.data(), a.data(), b.data());
  ASSERT_FALSE(didNotOverflow);
}