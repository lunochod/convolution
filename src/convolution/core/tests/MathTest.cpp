#include <convolution/core/logging.h>
#include <convolution/core/math.h>
#include <gtest/gtest.h>

using namespace convolution;

namespace {

template <class T>
class MathTestFixture : public testing::Test {
};

}  // namespace

typedef ::testing::Types<uint8_t> Implementations;
TYPED_TEST_SUITE(MathTestFixture, Implementations);

TYPED_TEST(MathTestFixture, AlignmentUnity) {
  constexpr TypeParam alignment = 1;
  const TypeParam size = 13;
  const TypeParam alignedSize = core::getAlignedSize<TypeParam, alignment>(size);
  ASSERT_EQ(alignedSize, size);
}

TYPED_TEST(MathTestFixture, AlignmentArbitrary) {
  constexpr TypeParam alignment = 7;
  const TypeParam size = 13;
  const TypeParam alignedSize = core::getAlignedSize<TypeParam, alignment>(size);
  ASSERT_EQ(14, alignedSize);
}

TYPED_TEST(MathTestFixture, AlignmentLargerThanSize) {
  constexpr TypeParam alignment = 7;
  const TypeParam size = 3;
  const TypeParam alignedSize = core::getAlignedSize<TypeParam, alignment>(size);
  ASSERT_EQ(alignment, alignedSize);
}
