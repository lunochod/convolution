#ifndef CONVOLUTION_CORE_TEST_TESTRESOURCES_H
#define CONVOLUTION_CORE_TEST_TESTRESOURCES_H

#include <cstdint>
#include <limits>
#include <random>
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

}  // namespace test
}  // namespace core
}  // namespace convolution

#endif  // CONVOLUTION_CORE_TESTS_TESTRESOURCES_H