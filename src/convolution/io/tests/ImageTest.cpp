// clang-format off
#include <gtest/gtest.h>
// clang-format on

#include <convolution/io/Image.h>
#include <convolution/core/Filter.h>
#include <convolution/core/logging.h>

#include <boost/preprocessor/stringize.hpp>

#include "CImg.h"

using namespace cimg_library;

using namespace convolution;

namespace {

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

TEST(ImageTest, TestImage) {
  const uint32_t imgWidth = 17;
  const uint32_t imgHeight = 13;
  auto img = createTestImage(imgHeight, imgWidth);

  uint32_t cnt = 0;
  for (int32_t y = 0; y < img.height(); ++y) {
    for (int32_t x = 0; x < img.width(); ++x) {
      ASSERT_EQ(img(x, y, 0, 0), cnt++);
    }
  }

  fs::path p = fs::path(std::string(BOOST_PP_STRINGIZE(PROJECT_SOURCE_DIR))) / "images" / "TestImage.bmp";
  img.save(p.c_str());
}

TEST(ImageTest, Read) {
  const uint32_t imgWidth = 17;
  const uint32_t imgHeight = 13;
  auto img = createTestImage(imgHeight, imgWidth);

  fs::path p = fs::path(std::string(BOOST_PP_STRINGIZE(PROJECT_SOURCE_DIR))) / "images" / "TestImage.bmp";
  img.save(p.c_str());

  io::Image image{};
  ASSERT_TRUE(image.read(p));
}
