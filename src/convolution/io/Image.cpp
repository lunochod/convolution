#include <convolution/core/math.h>
#include <convolution/io/Image.h>

#include <algorithm>

namespace convolution {
namespace io {

/// \brief Calculate an offset into the image buffer using image coordinates
///
/// The image buffer uses row-major format
///
/// \param ix (const uint32_t) x-position of the pixel in the image
/// \param iy (const uint32_t) y-position of the pixel in the image
/// \param ic (const uint32_t) channel of the pixel in the image
uint32_t Image::calcImageBufferOffset(const uint32_t ix, const uint32_t iy, const uint32_t channel) const {
  return pixels() * channel + width() * iy + ix;
}

/// \brief read image at the path provided into the image buffer
bool Image::read(const fs::path &path) {
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

template bool Image::img2col<core::MatrixOrder::kRowMajor>(const core::IFilter<uint8_t> &);
template bool Image::img2col<core::MatrixOrder::kColumnMajor>(const core::IFilter<uint8_t> &);

}  // namespace io
}  // namespace convolution