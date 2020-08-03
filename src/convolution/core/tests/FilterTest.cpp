#include <convolution/core/Filter.h>
#include <convolution/core/logging.h>
#include <convolution/core/tests/TestResources.h>
#include <gtest/gtest.h>

#include <limits>

using namespace convolution;

namespace {

template <class T>
class FilterTestFixture : public testing::Test {
};

}  // namespace

typedef ::testing::Types<uint8_t> Implementations;
TYPED_TEST_SUITE(FilterTestFixture, Implementations);

TYPED_TEST(FilterTestFixture, Instantiate) {
  constexpr uint32_t kHeight = 3;
  constexpr uint32_t kWidth = 5;

  auto elements = core::test::getRandomVector<TypeParam>(kHeight * kWidth);
  using TestFilter = core::Filter<TypeParam, kHeight, kWidth>;
  ASSERT_NO_THROW(TestFilter f(elements));
}

TYPED_TEST(FilterTestFixture, Get) {
  constexpr uint32_t kHeight = 3;
  constexpr uint32_t kWidth = 5;
  constexpr uint32_t kInputChannels = 3;
  constexpr uint32_t kOutputChannels = 1;

  auto elements = core::test::getRandomVector<TypeParam>(kHeight * kWidth * kInputChannels * kOutputChannels);
  using TestFilter = core::Filter<TypeParam, kHeight, kWidth, kInputChannels, kOutputChannels>;
  TestFilter f(elements);

  for (uint32_t ocIdx = 0; ocIdx < kOutputChannels; ++ocIdx) {
    for (uint32_t icIdx = 0; icIdx < kInputChannels; ++icIdx) {
      auto g = f.get(icIdx, ocIdx);
      const size_t offset = kHeight * kWidth * kInputChannels * ocIdx + kHeight * kWidth * icIdx;
      ASSERT_EQ(memcmp(g.getFilterBuffer(), f.getFilterBuffer() + offset, kWidth * kHeight * sizeof(TypeParam)), 0);
    }
  }
}

TYPED_TEST(FilterTestFixture, ReadAt) {
  constexpr uint32_t kHeight = 3;
  constexpr uint32_t kWidth = 5;
  constexpr uint32_t kInputChannels = 3;
  constexpr uint32_t kOutputChannels = 2;

  auto elements = core::test::getRandomVector<TypeParam>(kHeight * kWidth * kInputChannels * kOutputChannels);
  using TestFilter = core::Filter<TypeParam, kHeight, kWidth, kInputChannels, kOutputChannels>;
  TestFilter f(elements);

  for (uint32_t ocIdx = 0; ocIdx < kOutputChannels; ++ocIdx) {
    for (uint32_t icIdx = 0; icIdx < kInputChannels; ++icIdx) {
      for (uint32_t hIdx = 0; hIdx < kHeight; ++hIdx) {
        for (uint32_t wIdx = 0; wIdx < kWidth; ++wIdx) {
          const size_t offset = kHeight * kWidth * kInputChannels * ocIdx + kHeight * kWidth * icIdx + kHeight * hIdx + wIdx;
          ASSERT_EQ(f.getFilterBuffer()[offset], f.at(hIdx, wIdx, icIdx, ocIdx));
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

  auto elements = core::test::getRandomVector<TypeParam>(kHeight * kWidth * kInputChannels * kOutputChannels);
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

TYPED_TEST(FilterTestFixture, ColumnBuffer) {
  constexpr uint32_t kHeight = 3;
  constexpr uint32_t kWidth = 5;
  constexpr uint32_t kInputChannels = 3;
  constexpr uint32_t kOutputChannels = 2;

  using TestFilter = core::Filter<TypeParam, kHeight, kWidth, kInputChannels, kOutputChannels>;

  std::vector<TypeParam> data(TestFilter::kNumElements);
  for (uint32_t fIdx = 0; fIdx < data.size(); ++fIdx) {
    data[fIdx] = fIdx;
  }

  TestFilter filter(data);

  uint32_t cnt = 0;
  TypeParam *filterData = filter.getFilterBuffer();
  for (uint32_t oc = 0; oc < kOutputChannels; ++oc) {
    for (uint32_t ic = 0; ic < kInputChannels; ++ic) {
      for (uint32_t fy = 0; fy < kHeight; ++fy) {
        for (uint32_t fx = 0; fx < kWidth; ++fx) {
          uint32_t read = filter.calcFilterBufferOffset(fx, fy, ic, oc);
          ASSERT_EQ(filterData[read], cnt++);
        }
      }
    }
  }

  cnt = 0;
  TypeParam *columnData = filter.getColumnBuffer();
  for (uint32_t oc = 0; oc < kOutputChannels; ++oc) {
    for (uint32_t ic = 0; ic < kInputChannels; ++ic) {
      for (uint32_t fy = 0; fy < kHeight; ++fy) {
        for (uint32_t fx = 0; fx < kWidth; ++fx) {
          uint32_t read = filter.calcColumnBufferOffset(fx, fy, ic, oc);
          ASSERT_EQ(columnData[read], cnt++);
        }
      }
    }
  }
}