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

TEST(ImageTest, ColumnBuffer) {
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
  TestFilter filter;

  // use image reader to process test image and create column buffer
  io::Image image{};
  ASSERT_TRUE(image.read(p));

  const bool img2colRet = image.img2col<core::MatrixOrder::kRowMajor, alignment>(filter);
  ASSERT_TRUE(img2colRet);

  io::Image::StorageT colBuffer = *(image.getColumnBuffer());

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
        memset(patch.data(), 0, fHeight * fWidth);
        // read the image data covering the filter
        patchIdx = 0;
        for (uint32_t filter_y = 0; filter_y < filter.height(); ++filter_y) {
          for (uint32_t filter_x = 0; filter_x < filter.width(); ++filter_x, ++patchIdx) {
            // calculate the image coordinates to read from
            qx = img_x - filter.leftPadding() + filter_x;
            qy = img_y - filter.topPadding() + filter_y;
            if (qx >= 0 && qx < (int32_t)cimg.width()) {
              if (qy >= 0 && qy < (int32_t)cimg.height()) {
                // read pixel data into patch
                patch[patchIdx] = cimg(qx, qy, 0, img_c);
              }
            }
          }
        }

        uint8_t const* colPtr = colBuffer.data() + image.calcColumnBufferOffset<alignment>(filter, img_x, img_y, img_c, 0, 0);
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
