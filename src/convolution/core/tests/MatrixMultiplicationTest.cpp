#include <convolution/core/logging.h>
#include <convolution/core/math.h>
#include <convolution/core/tests/TestResources.h>
#include <gtest/gtest.h>

#include <limits>
#include <random>

using namespace convolution;

namespace {

template <class T>
class MatrixMultiplicationTestFixture : public testing::Test {
};

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
    std::fill(a.begin(), a.end(), 0);
    std::fill(b.begin(), b.end(), 1);
    std::fill(c.begin(), c.end(), 0);

    bool didNotOverflow = core::gemm<TypeParam, TypeParam, core::MatrixOrder::kRowMajor>(M, N, K, c.data(), a.data(), b.data());
    ASSERT_TRUE(didNotOverflow);

    for (uint32_t m = 0; m < M; ++m) {
      for (uint32_t n = 0; n < N; ++n) {
        ASSERT_EQ(c[core::address<core::MatrixOrder::kRowMajor>(M, N, m, n)], 0);
      }
    }
  }
  {
    std::fill(a.begin(), a.end(), 1);
    std::fill(b.begin(), b.end(), 0);
    std::fill(c.begin(), c.end(), 0);

    bool didNotOverflow = core::gemm<TypeParam, TypeParam, core::MatrixOrder::kRowMajor>(M, N, K, c.data(), a.data(), b.data());
    ASSERT_TRUE(didNotOverflow);

    for (uint32_t m = 0; m < M; ++m) {
      for (uint32_t n = 0; n < N; ++n) {
        ASSERT_EQ(c[core::address<core::MatrixOrder::kRowMajor>(M, N, m, n)], 0);
      }
    }
  }
}

TYPED_TEST(MatrixMultiplicationTestFixture, MulIdentity) {
  constexpr uint32_t M = 5;

  std::vector<TypeParam> a(M * M);
  std::vector<TypeParam> b(M * M);
  std::vector<TypeParam> c(M * M);

  core::test::initRandomMatrix<TypeParam>(M, M, a.data());
  core::test::initIdentityMatrix<TypeParam>(M, M, b.data());

  {
    std::fill(c.begin(), c.end(), 0);
    bool didNotOverflow = core::gemm<TypeParam, TypeParam, core::MatrixOrder::kRowMajor>(M, M, M, c.data(), a.data(), b.data());
    ASSERT_TRUE(didNotOverflow);

    for (uint32_t m = 0; m < M; ++m) {
      for (uint32_t n = 0; n < M; ++n) {
        ASSERT_EQ(c[m * M + n], a[m * M + n]);
      }
    }
  }
  {
    std::fill(c.begin(), c.end(), 0);
    bool didNotOverflow = core::gemm<TypeParam, TypeParam, core::MatrixOrder::kRowMajor>(M, M, M, c.data(), b.data(), a.data());
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

  core::test::initConstantMatrix<TypeParam>(M, M, a.data(), std::numeric_limits<TypeParam>::max());
  core::test::initConstantMatrix<TypeParam>(M, M, b.data(), std::numeric_limits<TypeParam>::max());
  std::fill(c.begin(), c.end(), 0);

  bool didNotOverflow = core::gemm<TypeParam, TypeParam, core::MatrixOrder::kRowMajor, true>(M, M, M, c.data(), a.data(), b.data());
  ASSERT_FALSE(didNotOverflow);
}

TYPED_TEST(MatrixMultiplicationTestFixture, kColumnMajor) {
  constexpr uint32_t M = 13;
  constexpr uint32_t N = 17;
  constexpr uint32_t K = 14;

  std::vector<TypeParam> a(M * K);
  std::vector<TypeParam> b(K * N);
  std::vector<TypeParam> c_test(M * N);
  std::vector<TypeParam> c_reference(M * N);

  core::test::initRandomMatrix<TypeParam>(M, K, a.data(), 1, 2);  //< init random matrix will values 1 or 2 to avoid overflow
  core::test::initRandomMatrix<TypeParam>(K, N, b.data(), 1, 2);  //< init random matrix will values 1 or 2 to avoid overflow

  std::fill(c_test.begin(), c_test.end(), 0);
  std::fill(c_reference.begin(), c_reference.end(), 0);

  // create the reference multiplication using core::gemm() where all matrices use kRowMajor
  bool referenceDidNotOverflow = core::gemm<TypeParam, TypeParam, core::MatrixOrder::kRowMajor, core::MatrixOrder::kRowMajor, core::MatrixOrder::kRowMajor, true>(M, N, K, c_reference.data(), a.data(), b.data());
  ASSERT_TRUE(referenceDidNotOverflow);

  // tranpose matrix a to be in kColumnMajor format
  std::vector<TypeParam> a_buffer(a.size());
  core::transpose<TypeParam, core::MatrixOrder::kRowMajor>(M, K, a.data(), a_buffer.data());

  // create the test multiplication using core::gemm() now using a in kColMajor format
  bool testDidNotOverflow = core::gemm<TypeParam, TypeParam, core::MatrixOrder::kRowMajor, core::MatrixOrder::kColumnMajor, core::MatrixOrder::kRowMajor, true>(M, N, K, c_test.data(), a.data(), b.data());
  ASSERT_TRUE(testDidNotOverflow);

  // the results of both multiplications must be identical
  ASSERT_EQ(memcmp(c_test.data(), c_reference.data(), c_test.size() * sizeof(TypeParam)), 0);
}

TYPED_TEST(MatrixMultiplicationTestFixture, HardwareMultiplier) {
  constexpr uint32_t M = 2;
  constexpr uint32_t N = 4;
  constexpr uint32_t K = 4;
  constexpr uint32_t P = 2;

  std::vector<TypeParam> a(M * K);
  std::vector<TypeParam> b(K * N);
  std::vector<TypeParam> c_test(M * N);
  std::vector<TypeParam> c_reference(M * N);

  core::test::initRandomMatrix<TypeParam>(M, K, a.data(), 1, 2);  //< init random matrix will values 1 or 2 to avoid overflow
  core::test::initRandomMatrix<TypeParam>(K, N, b.data(), 1, 2);  //< init random matrix will values 1 or 2 to avoid overflow

  std::fill(c_test.begin(), c_test.end(), 0);
  std::fill(c_reference.begin(), c_reference.end(), 0);

  // create the reference multiplication using core::gemm()
  bool referenceDidNotOverflow = core::gemm<TypeParam, TypeParam, core::MatrixOrder::kColumnMajor, core::MatrixOrder::kColumnMajor, core::MatrixOrder::kRowMajor, true>(M, N, K, c_reference.data(), a.data(), b.data());
  ASSERT_TRUE(referenceDidNotOverflow);

  // create the test multiplication using core::mult()
  bool testDidNotOverflow = core::mult<TypeParam, TypeParam, core::MatrixOrder::kColumnMajor, core::MatrixOrder::kColumnMajor, core::MatrixOrder::kRowMajor, P, true>(M, N, K, c_test.data(), a.data(), b.data());
  ASSERT_TRUE(testDidNotOverflow);

  ASSERT_EQ(memcmp(c_test.data(), c_reference.data(), c_test.size() * sizeof(TypeParam)), 0);
}
