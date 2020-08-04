#include <convolution/core/MatrixTranspose.h>
#include <convolution/core/logging.h>
#include <gtest/gtest.h>

#include <limits>
#include <random>

using namespace convolution;

namespace {

template <class T>
class MatrixTransposeTestFixture : public testing::Test {
};

template <typename T>
void initRandomMatrix(const uint32_t M, const uint32_t N, T *mat, const T min = 0, const T max = std::numeric_limits<T>::max()) {
  std::random_device device;
  std::mt19937 generator(device());
  std::uniform_int_distribution<T> distribution(min, max);

  for (uint32_t m = 0; m < M; ++m) {
    for (uint32_t n = 0; n < N; ++n) {
      mat[m * N + n] = distribution(generator);
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
TYPED_TEST_SUITE(MatrixTransposeTestFixture, Implementations);

TYPED_TEST(MatrixTransposeTestFixture, MN) {
  constexpr uint32_t M = 13;
  constexpr uint32_t N = 17;

  std::vector<TypeParam> a(M * N);
  std::vector<TypeParam> reference(M * N);
  std::vector<TypeParam> buffer(M * N);

  initRandomMatrix<TypeParam>(M, N, a.data());
  memcpy(reference.data(), a.data(), sizeof(TypeParam) * a.size());
  memset(buffer.data(), 0, buffer.size() * sizeof(TypeParam));

  core::transpose<TypeParam, core::MatrixOrder::kRowMajor>(M, N, a.data(), buffer.data());

  for (uint32_t m = 0; m < M; ++m) {
    for (uint32_t n = 0; n < N; ++n) {
      ASSERT_EQ(a[n * M + m], reference[m * N + n]);
    }
  }
}
