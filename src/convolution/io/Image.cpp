#include <convolution/core/math.h>
#include <convolution/io/Image.h>

#include <algorithm>

namespace convolution {
namespace io {

/// \brief Calculate an offset into the image buffer using image coordinates
/// \param img_x (const uint32_t) x-position of the pixel in the image
/// \param img_y (const uint32_t) y-position of the pixel in the image
/// \param img_c (const uint32_t) channel of the pixel in the image
/// \return (uint32_t) the offset into the image buffer to lookup the pixel data
uint32_t Image::calcImageBufferOffset(const uint32_t img_x, const uint32_t img_y, const uint32_t img_c) const {
  return pixels() * img_c + width() * img_y + img_x;
}

/// \brief read image at the path provided into the image buffer
/// \param path(const fs::path &) path to image on disk
/// \return true on success, false otherwise
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

/// \brief write selected channel of the image buffer to the path provided
/// \param path(const fs::path &) path on filesystem to write image
/// \param oc(const uint32_t) output channel to write
/// \return true on success, false otherwise
bool Image::write(const fs::path &path, const uint32_t oc) const {
  CImg<uint8_t> image(width(), height(), 1, 1);

  for (uint32_t img_y = 0; img_y < height(); ++img_y) {
    for (uint32_t img_x = 0; img_x < width(); ++img_x) {
      uint32_t read = calcImageBufferOffset(img_x, img_y, oc);
      image(img_x, img_y, 0, 0) = (*imgBufferPtr)[read];
    }
  }

  image.save(path.c_str());
  spdlog::info("Write image {} {}x{}x{} {} Byte", path.c_str(), width(), height(), 1, width() * height());
  return true;
}

}  // namespace io
}  // namespace convolution