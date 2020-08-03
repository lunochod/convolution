// clang-format off
#include <gtest/gtest.h>
// clang-format on

#include <convolution/core/Convolver.h>
#include <convolution/core/tests/TestResources.h>
#include <convolution/core/logging.h>

#include <limits>
#include <random>
#include <memory>
#include <cstdint>

using namespace convolution;

namespace {

}  // namespace

TEST(Convolution, Instantiate) {
  constexpr uint32_t kHeight = 3;
  constexpr uint32_t kWidth = 5;
  constexpr uint32_t kInputChannels = 3;
  constexpr uint32_t kOutputChannels = 2;

  auto elements = core::test::getRandomVector<uint8_t>(kHeight * kWidth * kInputChannels * kOutputChannels);
  using TestFilter = core::Filter<uint8_t, kHeight, kWidth, kInputChannels, kOutputChannels>;

  std::shared_ptr<TestFilter> filter = std::make_shared<TestFilter>(elements);

  core::Convolver conv(filter);

  fs::path p = "@CONVOLUTION_HOME@/images/Grace.jpg";
  conv(p);
}
