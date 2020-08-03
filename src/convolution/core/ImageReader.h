#ifndef CONVOLUTION_CORE_IMAGEREADER_H
#define CONVOLUTION_CORE_IMAGEREADER_H

#include <convolution/core/Filter.h>
#include <convolution/core/logging.h>

#include <cstdint>
#include <filesystem>
#include <memory>

#include "CImg.h"

using namespace cimg_library;
namespace fs = std::filesystem;

namespace convolution {
namespace core {

/// \class ImageReader
/// \brief Read and image from disk and convert it into column buffer format
class ImageReader {
 public:
  using StorageT = std::vector<uint8_t>;
  using StoragePtr = std::shared_ptr<StorageT>;

 private:
  uint32_t imgWidth = 0;              ///< image width in pixels
  uint32_t imgHeight = 0;             ///< image height in pixels
  uint32_t imgChannels = 0;           ///< number of image channels
  StoragePtr imgBufferPtr = nullptr;  ///< image buffer in row-major format
  StoragePtr colBufferPtr = nullptr;  ///< column buffer in row-major format

 public:
  bool read(const fs::path &path);               ///< read image at path into image buffer
  bool img2col(const IFilter<uint8_t> &filter);  ///< convert image in image buffer into column buffer

  uint32_t width() const { return imgWidth; };
  uint32_t height() const { return imgHeight; };
  uint32_t channels() const { return imgChannels; };
  uint32_t pixels() const { return imgWidth * imgHeight; };
  uint32_t elements() const { return imgWidth * imgHeight * imgChannels; }
  StoragePtr getColumnBuffer() const { return colBufferPtr; }

  /// address calculation into the image buffer
  uint32_t calcImageBufferOffset(const uint32_t ix, const uint32_t iy, const uint32_t channel) const;

  /// address calculation into the column buffer
  uint32_t calcColumnBufferOffset(const IFilter<uint8_t> &filter, const uint32_t ix, const uint32_t iy, const uint32_t ic, const uint32_t fx, const uint32_t fy) const;
};

}  // namespace core
}  // namespace convolution

#endif  // CONVOLUTION_CORE_IMAGEREADER_H