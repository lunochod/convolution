#include <convolution/core/Filter.h>
#include <convolution/core/logging.h>
#include <gtest/gtest.h>

#include <limits>
#include <random>

using namespace convolution;

namespace {

template <class T>
class FilterTestFixture : public testing::Test {
};

template <typename T, uint32_t kHeight, uint32_t kWidth, uint32_t kInputChannels = 1, uint32_t kOutputChannels = 1>
std::vector<T> initRandomFilterData() {
  constexpr uint32_t kNumElements = kHeight * kWidth * kInputChannels * kOutputChannels;
  static_assert(kNumElements != 0, "Filter dimensions are ill-defined.");

  std::random_device device;
  std::mt19937 generator(device());
  std::uniform_int_distribution<T> distribution(1, std::numeric_limits<T>::max());

  std::vector<T> elements(kNumElements);
  for (uint32_t idx = 0; idx < kNumElements; ++idx) {
    elements[idx] = distribution(generator);
  }

  return elements;  // return by value is ok since the vector data lives on the HEAP
}

}  // namespace

typedef ::testing::Types<uint8_t> Implementations;
TYPED_TEST_SUITE(FilterTestFixture, Implementations);

TYPED_TEST(FilterTestFixture, Instantiate) {
  constexpr uint32_t kHeight = 3;
  constexpr uint32_t kWidth = 5;

  auto elements = initRandomFilterData<TypeParam, kHeight, kWidth>();
  using TestFilter = core::Filter<TypeParam, kHeight, kWidth>;
  ASSERT_NO_THROW(TestFilter f(elements));
}

TYPED_TEST(FilterTestFixture, Get) {
  constexpr uint32_t kHeight = 3;
  constexpr uint32_t kWidth = 5;
  constexpr uint32_t kInputChannels = 3;
  constexpr uint32_t kOutputChannels = 1;

  auto elements = initRandomFilterData<TypeParam, kHeight, kWidth, kInputChannels, kOutputChannels>();
  using TestFilter = core::Filter<TypeParam, kHeight, kWidth, kInputChannels, kOutputChannels>;
  TestFilter f(elements);

  for (uint32_t ocIdx = 0; ocIdx < kOutputChannels; ++ocIdx) {
    for (uint32_t icIdx = 0; icIdx < kInputChannels; ++icIdx) {
      auto g = f.get(icIdx, ocIdx);
      const size_t offset = kHeight * kWidth * kInputChannels * ocIdx + kHeight * kWidth * icIdx;
      ASSERT_EQ(memcmp(g.data(), f.data() + offset, kWidth * kHeight * sizeof(TypeParam)), 0);
    }
  }
}

TYPED_TEST(FilterTestFixture, ReadAt) {
  constexpr uint32_t kHeight = 3;
  constexpr uint32_t kWidth = 5;
  constexpr uint32_t kInputChannels = 3;
  constexpr uint32_t kOutputChannels = 2;

  auto elements = initRandomFilterData<TypeParam, kHeight, kWidth, kInputChannels, kOutputChannels>();
  using TestFilter = core::Filter<TypeParam, kHeight, kWidth, kInputChannels, kOutputChannels>;
  TestFilter f(elements);

  for (uint32_t ocIdx = 0; ocIdx < kOutputChannels; ++ocIdx) {
    for (uint32_t icIdx = 0; icIdx < kInputChannels; ++icIdx) {
      for (uint32_t hIdx = 0; hIdx < kHeight; ++hIdx) {
        for (uint32_t wIdx = 0; wIdx < kWidth; ++wIdx) {
          const size_t offset = kHeight * kWidth * kInputChannels * ocIdx + kHeight * kWidth * icIdx + kHeight * hIdx + wIdx;
          ASSERT_EQ(f.data()[offset], f.at(hIdx, wIdx, icIdx, ocIdx));
        }
      }
    }
  }
}

TYPED_TEST(FilterTestFixture, WriteAt) {
  constexpr uint32_t kHeight = 3;
  constexpr uint32_t kWidth = 5;
  constexpr uint32_t kInputChannels = 3;
  constexpr uint32_t kOutputChannels = 2;

  auto elements = initRandomFilterData<TypeParam, kHeight, kWidth, kInputChannels, kOutputChannels>();
  using TestFilter = core::Filter<TypeParam, kHeight, kWidth, kInputChannels, kOutputChannels>;
  TestFilter f(elements);

  uint32_t cnt = 0;
  for (uint32_t ocIdx = 0; ocIdx < kOutputChannels; ++ocIdx) {
    for (uint32_t icIdx = 0; icIdx < kInputChannels; ++icIdx) {
      for (uint32_t hIdx = 0; hIdx < kHeight; ++hIdx) {
        for (uint32_t wIdx = 0; wIdx < kWidth; ++wIdx) {
          f.at(hIdx, wIdx, icIdx, ocIdx) = cnt;
          ASSERT_EQ(f.at(hIdx, wIdx, icIdx, ocIdx), cnt);
          ++cnt;
        }
      }
    }
  }
}
