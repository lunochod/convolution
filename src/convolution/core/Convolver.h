#ifndef CONVOLUTION_CORE_CONVOLVER_H
#define CONVOLUTION_CORE_CONVOLVER_H

#include <convolution/core/Filter.h>
#include <convolution/core/logging.h>
#include <convolution/io/Image.h>

#include <vector>

namespace convolution {
namespace core {

/// \class A class to convolve 8Bit image data using an 8Bit 4D filter
/// To avoid overflow the results of the convolution are stored in a 16Bit data format
/// \tparam alignment(uint32_t) align the column and filter buffer in support of the MxPxP multiplier to be used
template <uint32_t alignment>
class Convolver {
 public:
  using ColumnDataT = uint8_t;
  using ColumnBufferT = std::vector<ColumnDataT>;
  using ColumnBufferPtr = std::shared_ptr<ColumnBufferT>;

  using TransformDataT = uint16_t;
  using TransformBufferT = std::vector<TransformDataT>;
  using TransformBufferPtr = std::shared_ptr<TransformBufferT>;

 private:
  ColumnBufferPtr colBufferPtr = std::make_shared<ColumnBufferT>();              ///< the column buffer is used to store the result of transforming the image into column format
  TransformBufferPtr transformBufferPtr = std::make_shared<TransformBufferT>();  ///< the transform buffer is used to store the results of multiplying the column buffer with the filter
  std::shared_ptr<IFilter<ColumnDataT>> filterPtr;                               ///< filter used for the convolution
  io::Image img;                                                                 ///< image used for the convolution

 protected:
  /// convert image into column buffer format
  template <core::MatrixOrder order = core::MatrixOrder::kRowMajor>
  bool img2col();

  /// read image located at path
  bool read(const fs::path &path);

  /// address calculation into the column buffer
  uint32_t calcColumnBufferOffset(const uint32_t ix, const uint32_t iy, const uint32_t ic, const uint32_t fx, const uint32_t fy) const;

  ColumnBufferPtr getColumnBuffer() const;
  TransformBufferPtr getTransformBuffer() const;

 public:
  explicit Convolver(std::shared_ptr<IFilter<uint8_t>> f) : filterPtr(f), img() {}
  void operator()(const fs::path &path);
};

}  // namespace core
}  // namespace convolution

#include <convolution/core/Convolver.inl>

#endif  // CONVOLUTION_CORE_CONVOLVER_H