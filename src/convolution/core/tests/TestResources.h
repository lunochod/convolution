#ifndef CONVOLUTION_CORE_TEST_TESTRESOURCES_H
#define CONVOLUTION_CORE_TEST_TESTRESOURCES_H

#include <convolution/core/math.h>

#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <vector>

namespace convolution {
namespace core {
namespace test {

template <typename T>
std::vector<T> getRandomVector(const uint32_t numElements) {
  std::random_device device;
  std::mt19937 generator(device());
  std::uniform_int_distribution<T> distribution(1, std::numeric_limits<T>::max());

  std::vector<T> elements(numElements);
  for (uint32_t idx = 0; idx < numElements; ++idx) {
    elements[idx] = distribution(generator);
  }

  return elements;  // return by value is ok since the vector data lives on the HEAP
}

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

template <typename T, core::MatrixOrder order>
void print(const uint32_t M, const uint32_t N, const T *data) {
  std::stringstream ss;
  for (uint32_t m = 0; m < M; ++m) {
    for (uint32_t n = 0; n < N; ++n) {
      ss << data[core::address<order>(M, N, m, n)];
    }
    ss << std::endl;
  }
  std::cout << ss.str() << std::endl;
}

}  // namespace test
}  // namespace core
}  // namespace convolution

#endif  // CONVOLUTION_CORE_TESTS_TESTRESOURCES_H