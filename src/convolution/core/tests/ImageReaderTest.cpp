// clang-format off
#include <gtest/gtest.h>
// clang-format on

#include <convolution/core/ImageReader.h>
#include <convolution/core/Filter.h>
#include <convolution/core/logging.h>

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

TEST(ImageReaderTest, TestImage) {
  const uint32_t imgWidth = 17;
  const uint32_t imgHeight = 13;
  auto img = createTestImage(imgHeight, imgWidth);

  uint32_t cnt = 0;
  for (int32_t y = 0; y < img.height(); ++y) {
    for (int32_t x = 0; x < img.width(); ++x) {
      ASSERT_EQ(img(x, y, 0, 0), cnt++);
    }
  }

  img.save("TestImage.bmp");
}

TEST(ImageReaderTest, Read) {
  const uint32_t imgWidth = 17;
  const uint32_t imgHeight = 13;
  auto img = createTestImage(imgHeight, imgWidth);
  img.save("TestImage.bmp");

  core::ImageReader image{};
  ASSERT_TRUE(image.read("TestImage.bmp"));
}

TEST(ImageReaderTest, ColumnBuffer) {
  // setup test image
  const uint32_t imgWidth = 17;
  const uint32_t imgHeight = 13;
  auto img = createTestImage(imgHeight, imgWidth);
  img.save("TestImage.bmp");

  // setup test filter
  constexpr uint32_t fHeight = 3;
  constexpr uint32_t fWidth = 5;
  constexpr uint32_t kInputChannels = 3;
  constexpr uint32_t kOutputChannels = 2;
  using TestFilter = core::Filter<uint8_t, fHeight, fWidth, kInputChannels, kOutputChannels>;
  TestFilter filter;

  // use image reader to process test image and create column buffer
  core::ImageReader image{};
  ASSERT_TRUE(image.read("TestImage.bmp"));
  ASSERT_TRUE(image.img2col(filter));
  core::ImageReader::StorageT colBuffer = *(image.getColumnBuffer());

  // read the test image directly to create the comparison data
  CImg<uint8_t> cimg("TestImage.bmp");

  // the patch will contain the image data that covers the filter area
  core::ImageReader::StorageT patch(fHeight * fWidth);
  uint32_t patchIdx = 0;
  int32_t qx = 0;
  int32_t qy = 0;

  // iterate over the test image line-by-line vertically
  for (int32_t iy = 0; iy < cimg.height(); ++iy) {
    // iterate over the current line pixel-by-pixel horizontally
    for (int32_t ix = 0; ix < cimg.width(); ++ix) {
      // clear the patch
      memset(patch.data(), 0, fHeight * fWidth);
      // read the image data covering the filter
      patchIdx = 0;
      for (uint32_t fy = 0; fy < filter.height(); ++fy) {
        for (uint32_t fx = 0; fx < filter.width(); ++fx, ++patchIdx) {
          // calculate the image coordinates to read from
          qx = ix - filter.leftPadding() + fx;
          qy = iy - filter.topPadding() + fy;
          if (qx >= 0 && qx < (int32_t)cimg.width()) {
            if (qy >= 0 && qy < (int32_t)cimg.height()) {
              // read pixel data into patch
              patch[patchIdx] = cimg(qx, qy, 0, 0);
            }
          }
        }
      }

      uint8_t const* colPtr = colBuffer.data() + image.calcColumnBufferOffset(filter, ix, iy, 0, 0, 0);
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
