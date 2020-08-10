#ifndef CONVOLUTION_CORE_CONVOLVER_H
#define CONVOLUTION_CORE_CONVOLVER_H

#include <convolution/core/Filter.h>
#include <convolution/core/logging.h>
#include <convolution/io/Image.h>

#include <vector>

namespace convolution {
namespace core {

template <uint32_t alignment>
class Convolver {
 public:
  using StorageT = std::vector<uint8_t>;
  using StoragePtr = std::shared_ptr<StorageT>;

 private:
  std::shared_ptr<IFilter<uint8_t>> filterPtr;
  io::Image img;
  StoragePtr colBufferPtr = std::make_shared<StorageT>();        ///< column buffer
  StoragePtr transformBufferPtr = std::make_shared<StorageT>();  ///< transform buffer

 protected:
  template <core::MatrixOrder order = core::MatrixOrder::kRowMajor>
  bool img2col();

  bool read(const fs::path &path) {
    return img.read(path);
  }

  /// address calculation into the column buffer
  uint32_t calcColumnBufferOffset(const uint32_t ix, const uint32_t iy, const uint32_t ic, const uint32_t fx, const uint32_t fy) const;

  StoragePtr getColumnBuffer() const;
  StoragePtr getTransformBuffer() const;

 public:
  explicit Convolver(std::shared_ptr<IFilter<uint8_t>> f) : filterPtr(f), img() {}
  void operator()(const fs::path &path);
};

}  // namespace core
}  // namespace convolution

#include <convolution/core/Convolver.inl>

#endif  // CONVOLUTION_CORE_CONVOLVER_H