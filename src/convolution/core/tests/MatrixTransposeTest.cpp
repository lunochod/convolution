#include <convolution/core/logging.h>
#include <convolution/core/math.h>
#include <convolution/core/tests/TestResources.h>
#include <gtest/gtest.h>

#include <limits>
#include <random>

using namespace convolution;

namespace {

template <class T>
class MatrixTransposeTestFixture : public testing::Test {
};

}  // namespace

typedef ::testing::Types<uint8_t> Implementations;
TYPED_TEST_SUITE(MatrixTransposeTestFixture, Implementations);

TYPED_TEST(MatrixTransposeTestFixture, MN) {
  constexpr uint32_t M = 13;
  constexpr uint32_t N = 17;

  std::vector<TypeParam> a(M * N);
  std::vector<TypeParam> reference(M * N);
  std::vector<TypeParam> buffer(M * N);

  core::test::initRandomMatrix<TypeParam>(M, N, a.data());
  memcpy(reference.data(), a.data(), sizeof(TypeParam) * a.size());
  memset(buffer.data(), 0, buffer.size() * sizeof(TypeParam));

  core::transpose<TypeParam, core::MatrixOrder::kRowMajor>(M, N, a.data(), buffer.data());

  for (uint32_t m = 0; m < M; ++m) {
    for (uint32_t n = 0; n < N; ++n) {
      ASSERT_EQ(a[n * M + m], reference[m * N + n]);
    }
  }
}
