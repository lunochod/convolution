// clang-format off
#include <gtest/gtest.h>
// clang-format on

#include <convolution/core/Convolver.h>
#include <convolution/core/tests/TestResources.h>
#include <convolution/core/logging.h>

#include <boost/preprocessor/stringize.hpp>

#include <limits>
#include <random>
#include <memory>
#include <cstdint>
#include <string>
#include <cstdlib>
#include <sstream>

using namespace convolution;

namespace {

}  // namespace

TEST(Convolution, IdentityFilter) {
  constexpr uint32_t P = 2;  //< hardware multiplier size
  constexpr uint32_t kHeight = 1;
  constexpr uint32_t kWidth = 1;
  constexpr uint32_t kInputChannels = 3;
  constexpr uint32_t kOutputChannels = 3;

  // 1x1 filter picking the red, green and blue channel
  // clang-format off
  std::vector<uint8_t> elements = {
      1, 0, 0, // red
      0, 1, 0, // green
      0, 0, 1  // blue
  };
  // clang-format on

  using TestFilter = core::Filter<uint8_t, kHeight, kWidth, kInputChannels, kOutputChannels, P>;

  // create a TestFilter using random data for the filter elements
  std::shared_ptr<TestFilter> filter = std::make_shared<TestFilter>(elements);

  // create the Convolver using the previously defined filter
  core::Convolver<P> conv(filter);

  // define a test image
  fs::path inputFile = fs::path(std::string(BOOST_PP_STRINGIZE(PROJECT_SOURCE_DIR))) / "images" / "Grace.jpg";
  fs::path redFile = fs::path(std::string(BOOST_PP_STRINGIZE(PROJECT_SOURCE_DIR))) / "images" / "Grace_0.png";
  fs::path greenFile = fs::path(std::string(BOOST_PP_STRINGIZE(PROJECT_SOURCE_DIR))) / "images" / "Grace_1.png";
  fs::path blueFile = fs::path(std::string(BOOST_PP_STRINGIZE(PROJECT_SOURCE_DIR))) / "images" / "Grace_2.png";

  // apply convolution to image
  ASSERT_NO_THROW(conv(inputFile));

  // read original image
  io::Image original{};
  ASSERT_TRUE(original.read(inputFile));
  uint8_t *originalBuffer = original.getImageBuffer()->data();

  // read and compare red image
  io::Image red{};
  ASSERT_TRUE(red.read(redFile));
  uint8_t *redBuffer = red.getImageBuffer()->data();
  ASSERT_EQ(memcmp(originalBuffer, redBuffer, original.height() * original.width()), 0);

  // read and compare green image
  io::Image green{};
  ASSERT_TRUE(green.read(greenFile));
  uint8_t *greenBuffer = green.getImageBuffer()->data();
  ASSERT_EQ(memcmp(originalBuffer + original.height() * original.width(), greenBuffer, original.height() * original.width()), 0);

  // read and compare blue image
  io::Image blue{};
  ASSERT_TRUE(blue.read(blueFile));
  uint8_t *blueBuffer = blue.getImageBuffer()->data();
  ASSERT_EQ(memcmp(originalBuffer + 2 * original.height() * original.width(), blueBuffer, original.height() * original.width()), 0);
}
