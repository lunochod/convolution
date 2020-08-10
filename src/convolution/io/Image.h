#ifndef CONVOLUTION_IO_IMAGE_H
#define CONVOLUTION_IO_IMAGE_H

#include <convolution/core/Filter.h>
#include <convolution/core/logging.h>
#include <convolution/core/math.h>

#include <cstdint>
#include <filesystem>
#include <memory>

#include "CImg.h"

using namespace cimg_library;
namespace fs = std::filesystem;

namespace convolution {
namespace io {

/// \class Image class to support reading and writing images from and to disk
class Image {
 public:
  using StorageT = std::vector<uint8_t>;
  using StoragePtr = std::shared_ptr<StorageT>;

 private:
  uint32_t imgWidth = 0;              ///< image width in pixels
  uint32_t imgHeight = 0;             ///< image height in pixels
  uint32_t imgChannels = 0;           ///< number of image channels
  StoragePtr imgBufferPtr = nullptr;  ///< image buffer in row-major format

 public:
  bool read(const fs::path &path);                            ///< read image at path into image buffer
  bool write(const fs::path &path, const uint32_t oc) const;  ///< write transform buffer to image specified at path

  uint32_t width() const { return imgWidth; };
  uint32_t height() const { return imgHeight; };
  uint32_t channels() const { return imgChannels; };
  uint32_t pixels() const { return imgWidth * imgHeight; };
  uint32_t elements() const { return imgWidth * imgHeight * imgChannels; }

  StoragePtr getImageBuffer() const { return imgBufferPtr; }

  /// address calculation into the image buffer
  uint32_t calcImageBufferOffset(const uint32_t ix, const uint32_t iy, const uint32_t channel) const;
};

}  // namespace io
}  // namespace convolution

#endif  // CONVOLUTION_IO_IMAGE_H