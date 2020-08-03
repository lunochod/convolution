#include <convolution/core/ImageReader.h>

namespace convolution {
namespace core {

/// \brief Calculate an offset into the column buffer using image and filter coordinates
/// The layout of the column buffer format follows: http://15418.courses.cs.cmu.edu/fall2017/lecture/dnn/slide_023
///
/// The image, filter and column buffer all use row-major format
///
/// \param filter (const IFilter &) reference to the filter being used
/// \param ix (const uint32_t) x-position of the pixel in the image
/// \param iy (const uint32_t) y-position of the pixel in the image
/// \param ic (const uint32_t) channel of the pixel in the image
/// \param fx (const uint32_t) x-position in the filter
/// \param fy (const uint32_t) y-position in the filter
uint32_t ImageReader::calcColumnBufferOffset(const IFilter &filter, const uint32_t ix, const uint32_t iy, const uint32_t ic, const uint32_t fx, const uint32_t fy) const {
  const uint32_t pixelIndex = width() * iy + ix;
  const uint32_t filterSize = filter.width() * filter.height();
  return pixelIndex * filterSize * channels() + ic * filterSize + filter.width() * fy + fx;
}

/// \brief Calculate an offset into the image buffer using image coordinates
///
/// The image buffer uses row-major format
///
/// \param ix (const uint32_t) x-position of the pixel in the image
/// \param iy (const uint32_t) y-position of the pixel in the image
/// \param ic (const uint32_t) channel of the pixel in the image
uint32_t ImageReader::calcImageBufferOffset(const uint32_t ix, const uint32_t iy, const uint32_t channel) const {
  return pixels() * channel + width() * iy + ix;
}

/// \brief read image at the path provided into the image buffer
bool ImageReader::read(const fs::path &path) {
  if (!fs::exists(path)) {
    spdlog::error("File {} doesn't exist.", path.c_str());
    return false;
  }

  CImg<uint8_t> image(path.c_str());
  imgWidth = image.width();
  imgHeight = image.height();
  imgChannels = image.spectrum();

  const uint32_t numElements = imgWidth * imgHeight * imgChannels;
  imgBufferPtr = std::make_shared<StorageT>(numElements);
  StorageT &imgBuffer = *imgBufferPtr;
  memcpy(imgBuffer.data(), image.data(), numElements);

  spdlog::info("Read image {} {}x{}x{} {} Byte", path.c_str(), width(), height(), channels(), imgBuffer.size());
  return true;
}

/// \brief convert image to column buffer format
/// The layout of the column buffer format follows: http://15418.courses.cs.cmu.edu/fall2017/lecture/dnn/slide_023
///
/// The number of memory reads and writes required to complete the operation:
///   filterWidth x filterHeight x numChannels x imageWidth x imageHeight
///
/// The innermost loop copies the same data of size filterWidth filterHeight times, this small amount of data
/// is kept in the CPU cache, so that the number of DRAM reads is at most:
///   filterWidth x numChannels x imageWidth x imageHeight
///
/// Since we read the image data line-by-line and iterate pixel-by-pixel the overlapping reads of filterWidth
/// will be cached in the CPU cache, so that the number of DRAM reads is further reduced to:
///   numChannels x imageWidth x imageHeight
///
/// Read Complexity  : O( sizeof(Image) )
/// Write Complexity : O( sizeof(Image) x sizeof(Filter) )
///
bool ImageReader::img2col(const IFilter &filter) {
  if (imgBufferPtr->empty()) {
    spdlog::error("Image buffer is empty, failed to initialize column buffer.");
    return false;
  }

  const uint32_t imgWidth = width();
  const uint32_t imgHeight = height();
  const uint32_t imgChannels = channels();
  const uint32_t filterWidth = filter.width();
  const uint32_t filterHeight = filter.height();
  const uint32_t paddingWidth = filter.leftPadding();
  const uint32_t paddingHeight = filter.topPadding();

  // create a clear a line buffer with sufficient space for left and right padding
  StorageT lineBuffer(imgWidth + filterWidth - 1);
  memset(lineBuffer.data(), 0, lineBuffer.size());

  // resize and clear the column buffer
  colBufferPtr.reset();
  colBufferPtr = std::make_shared<StorageT>(filterWidth * filterHeight * pixels() * channels());
  StorageT &colBuffer = *colBufferPtr;
  memset(colBuffer.data(), 0, colBuffer.size());

  // iterate over each channel
  for (uint32_t img_c = 0; img_c < imgChannels; ++img_c) {
    // iterate over image line-by-line vertically
    for (uint32_t img_y = 0; img_y < imgHeight; ++img_y) {
      // copy current image line into the lineBuffer and add horizontal padding
      uint32_t imgOffset = calcImageBufferOffset(0, img_y, img_c);
      memcpy(lineBuffer.data() + paddingWidth, imgBufferPtr->data() + imgOffset, imgWidth);
      auto bgn = lineBuffer.begin();
      // iterate over the current image line pixel-by-pixel horizontally
      for (uint32_t img_x = 0; img_x < imgWidth; ++img_x, ++bgn) {
        // iterate vertical over the filter
        for (uint32_t filter_y = 0; filter_y < filterHeight; ++filter_y) {
          // each filter_y position corresponds to a single copy of the data at position bgn into the column buffer
          // calculate the destination line in the image for the current copy
          const int32_t dst_y = img_y - paddingHeight + filter_y;
          if (dst_y >= 0 && dst_y < (int32_t)imgHeight) {
            auto wIt = colBuffer.begin() + calcColumnBufferOffset(filter, img_x, dst_y, img_c, 0, filterHeight - filter_y - 1);
            std::copy(bgn, bgn + filterWidth, wIt);
          }
        }
      }
    }
  }

  return true;
}

}  // namespace core
}  // namespace convolution