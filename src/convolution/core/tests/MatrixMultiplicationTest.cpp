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
    memset(a.data(), 0, a.size() * sizeof(TypeParam));
    memset(b.data(), 1, b.size() * sizeof(TypeParam));
    memset(c.data(), 0, c.size() * sizeof(TypeParam));

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
    memset(c.data(), 0, c.size() * sizeof(TypeParam));

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

  core::test::initRandomMatrix<TypeParam>(M, M, a.data());
  core::test::initIdentityMatrix<TypeParam>(M, M, b.data());

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

  core::test::initConstantMatrix<TypeParam>(M, M, a.data(), std::numeric_limits<TypeParam>::max());
  core::test::initConstantMatrix<TypeParam>(M, M, b.data(), std::numeric_limits<TypeParam>::max());
  memset(c.data(), 0, c.size() * sizeof(TypeParam));
  bool didNotOverflow = core::gemm<TypeParam, core::MatrixOrder::kRowMajor, true>(M, M, M, c.data(), a.data(), b.data());
  ASSERT_FALSE(didNotOverflow);
}

TYPED_TEST(MatrixMultiplicationTestFixture, kColumnMajor) {
  constexpr uint32_t M = 3;  //13;
  constexpr uint32_t N = 4;  //17;
  constexpr uint32_t K = 5;  //14;

  std::vector<TypeParam> a(M * K);
  std::vector<TypeParam> b(K * N);
  std::vector<TypeParam> c_test(M * N);
  std::vector<TypeParam> c_reference(M * N);

  core::test::initRandomMatrix<TypeParam>(M, K, a.data(), 1, 2);  //< init random matrix will values 1 or 2 to avoid overflow
  core::test::initRandomMatrix<TypeParam>(K, N, b.data(), 1, 2);  //< init random matrix will values 1 or 2 to avoid overflow

  memset(c_test.data(), 0, c_test.size() * sizeof(TypeParam));
  memset(c_reference.data(), 0, c_reference.size() * sizeof(TypeParam));

  // create the reference multiplication using core::gemm()
  bool referenceDidNotOverflow = core::gemm<TypeParam, core::MatrixOrder::kRowMajor, core::MatrixOrder::kRowMajor, core::MatrixOrder::kRowMajor, true>(M, N, K, c_reference.data(), a.data(), b.data());
  ASSERT_TRUE(referenceDidNotOverflow);

  // tranpose a
  std::vector<TypeParam> a_buffer(a.size());
  core::transpose<TypeParam, core::MatrixOrder::kRowMajor>(M, K, a.data(), a_buffer.data());

  // create the test multiplication using core::mult()
  bool testDidNotOverflow = core::gemm<TypeParam, core::MatrixOrder::kRowMajor, core::MatrixOrder::kColumnMajor, core::MatrixOrder::kRowMajor, true>(M, N, K, c_test.data(), a.data(), b.data());
  ASSERT_TRUE(testDidNotOverflow);

  ASSERT_EQ(memcmp(c_test.data(), c_reference.data(), c_test.size() * sizeof(TypeParam)), 0);
}

TYPED_TEST(MatrixMultiplicationTestFixture, HardwareMultiplierMulIdentity) {
  constexpr uint32_t M = 13;
  constexpr uint32_t N = 17;
  constexpr uint32_t K = 14;
  constexpr uint32_t P = 4;

  std::vector<TypeParam> a(M * K);
  std::vector<TypeParam> b(K * N);
  std::vector<TypeParam> c_test(M * N);
  std::vector<TypeParam> c_reference(M * N);

  core::test::initRandomMatrix<TypeParam>(M, K, a.data(), 1, 2);  //< init random matrix will values 1 or 2 to avoid overflow
  core::test::initRandomMatrix<TypeParam>(K, N, b.data(), 1, 2);  //< init random matrix will values 1 or 2 to avoid overflow

  memset(c_test.data(), 0, c_test.size() * sizeof(TypeParam));
  memset(c_reference.data(), 0, c_reference.size() * sizeof(TypeParam));

  // create the reference multiplication using core::gemm()
  bool referenceDidNotOverflow = core::gemm<TypeParam, core::MatrixOrder::kRowMajor, core::MatrixOrder::kColumnMajor, core::MatrixOrder::kRowMajor, true>(M, N, K, c_reference.data(), a.data(), b.data());
  ASSERT_TRUE(referenceDidNotOverflow);

  // create the test multiplication using core::mult()
  bool testDidNotOverflow = core::mult<TypeParam, core::MatrixOrder::kRowMajor, core::MatrixOrder::kColumnMajor, core::MatrixOrder::kRowMajor, P, true>(M, N, K, c_test.data(), a.data(), b.data());
  ASSERT_TRUE(testDidNotOverflow);

  ASSERT_EQ(memcmp(c_test.data(), c_reference.data(), c_test.size() * sizeof(TypeParam)), 0);
}
