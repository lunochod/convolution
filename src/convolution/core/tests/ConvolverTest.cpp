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

#include "CImg.h"

using namespace cimg_library;
namespace fs = std::filesystem;

using namespace convolution;

namespace {

template <uint32_t alignment>
class TestConvolver : public core::Convolver<alignment> {
 public:
  TestConvolver(std::shared_ptr<core::IFilter<uint8_t>> f) : core::Convolver<alignment>(f) {}
  using core::Convolver<alignment>::img2col;
  using core::Convolver<alignment>::read;
  using core::Convolver<alignment>::calcColumnBufferOffset;
  using core::Convolver<alignment>::getColumnBuffer;
};

CImg<uint8_t> createTestImage(uint32_t height, uint32_t width) {
  CImg<uint8_t> img(width, height, 1, 3);

  uint32_t cnt = 0;
  for (uint32_t y = 0; y < height; ++y) {
    for (uint32_t x = 0; x < width; ++x) {
      *(img.data(x, y, 0, 0)) = cnt++;
      *(img.data(x, y, 0, 1)) = img(x, y, 0, 0);
      *(img.data(x, y, 0, 2)) = img(x, y, 0, 0);
    }
  }
  return img;
}

}  // namespace

TEST(ConvolverTest, ColumnBuffer) {
  // setup test image
  const uint32_t imgWidth = 17;
  const uint32_t imgHeight = 13;
  auto img = createTestImage(imgHeight, imgWidth);

  fs::path p = fs::path(std::string(BOOST_PP_STRINGIZE(PROJECT_SOURCE_DIR))) / "images" / "TestImage.bmp";
  img.save(p.c_str());

  // setup test filter
  constexpr uint32_t fHeight = 3;
  constexpr uint32_t fWidth = 5;
  constexpr uint32_t kInputChannels = 3;
  constexpr uint32_t kOutputChannels = 2;
  constexpr uint32_t alignment = 3;
  using TestFilter = core::Filter<uint8_t, fHeight, fWidth, kInputChannels, kOutputChannels, alignment>;

  // create a TestFilter
  std::shared_ptr<TestFilter> filter = std::make_shared<TestFilter>();

  // create the Convolver using the previously defined filter
  TestConvolver<alignment> conv(filter);

  // use convolver to read image
  ASSERT_TRUE(conv.read(p));

  // apply img2col transform to image
  ASSERT_TRUE(conv.img2col<core::MatrixOrder::kRowMajor>());

  // get the column buffer for inspection
  io::Image::StorageT colBuffer = *(conv.getColumnBuffer());

  // read the test image directly to create the comparison data
  CImg<uint8_t> cimg(p.c_str());

  // the patch will contain the image data that covers the filter area
  io::Image::StorageT patch(fHeight * fWidth);
  uint32_t patchIdx = 0;
  int32_t qx = 0;
  int32_t qy = 0;

  // iterate over the test image channel-by-channel
  for (int32_t img_c = 0; img_c < cimg.spectrum(); ++img_c) {
    // iterate over the test image line-by-line vertically
    for (int32_t img_y = 0; img_y < cimg.height(); ++img_y) {
      // iterate over the current line pixel-by-pixel horizontally
      for (int32_t img_x = 0; img_x < cimg.width(); ++img_x) {
        // clear the patch
        std::fill(patch.begin(), patch.end(), 0);
        // read the image data covering the filter
        patchIdx = 0;
        for (uint32_t filter_y = 0; filter_y < filter->height(); ++filter_y) {
          for (uint32_t filter_x = 0; filter_x < filter->width(); ++filter_x, ++patchIdx) {
            // calculate the image coordinates to read from
            qx = img_x - filter->leftPadding() + filter_x;
            qy = img_y - filter->topPadding() + filter_y;
            if (qx >= 0 && qx < (int32_t)cimg.width()) {
              if (qy >= 0 && qy < (int32_t)cimg.height()) {
                // read pixel data into patch
                patch[patchIdx] = cimg(qx, qy, 0, img_c);
              }
            }
          }
        }

        uint8_t const *colPtr = colBuffer.data() + conv.calcColumnBufferOffset(img_x, img_y, img_c, 0, 0);
        // compare the patch data with the data in the column buffer
        if (memcmp(patch.data(), colPtr, patch.size()) != 0) {
          printf("TEST: ");
          for (uint32_t pIdx = 0; pIdx < patch.size(); ++pIdx) {
            printf("%4d", patch[pIdx]);
          }
          printf("\n");
          printf("COLB: ");
          for (uint32_t pIdx = 0; pIdx < patch.size(); ++pIdx) {
            printf("%4d", colPtr[pIdx]);
          }
          printf("\n");
        }
        ASSERT_EQ(memcmp(patch.data(), colPtr, patch.size()), 0);
      }
    }
  }
}

TEST(Convolution, ColorFilter) {
  constexpr uint32_t P = 8;
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
