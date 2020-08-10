#ifndef CONVOLUTION_CORE_CONVOLVER_H
#define CONVOLUTION_CORE_CONVOLVER_H

#include <convolution/core/Filter.h>
#include <convolution/core/logging.h>
#include <convolution/io/Image.h>

#include <vector>

namespace convolution {
namespace core {

/// \class Convolver
/// \brief A class to convolve 8Bit image data with an 8Bit 4D filter using a 16Bit accumulator
/// \tparam alignment(uint32_t) specifies the alignment of the column and filter buffer in support of the MxPxP multiplier to be used
template <uint32_t alignment>
class Convolver {
 public:
  using ColumnDataT = uint8_t;                             ///< the C++ type used to represent a single input channel pixel
  using ColumnBufferT = std::vector<ColumnDataT>;          ///< the storage format used by the input column buffer
  using ColumnBufferPtr = std::shared_ptr<ColumnBufferT>;  ///< shared pointer to the column buffer

  using TransformDataT = uint16_t;                               ///< the C++ type used to represent a single output channel pixel
  using TransformBufferT = std::vector<TransformDataT>;          ///< the storage format used by the output transform buffer
  using TransformBufferPtr = std::shared_ptr<TransformBufferT>;  ///< shared pointer to the transform buffer

 private:
  ColumnBufferPtr colBufferPtr = std::make_shared<ColumnBufferT>();              ///< the column buffer is used to store the result of transforming the image into column format
  TransformBufferPtr transformBufferPtr = std::make_shared<TransformBufferT>();  ///< the transform buffer is used to store the results of multiplying the column buffer with the filter
  std::shared_ptr<IFilter<ColumnDataT>> filterPtr;                               ///< filter used for the convolution
  io::Image img;                                                                 ///< image used for the convolution

 protected:
  template <core::MatrixOrder order = core::MatrixOrder::kRowMajor>
  bool img2col();

  bool read(const fs::path &path);

  uint32_t calcColumnBufferOffset(const uint32_t ix, const uint32_t iy, const uint32_t ic, const uint32_t fx, const uint32_t fy) const;

  ColumnBufferPtr getColumnBuffer() const;
  TransformBufferPtr getTransformBuffer() const;

 public:
  explicit Convolver(std::shared_ptr<IFilter<ColumnDataT>> f) : filterPtr(f), img() {}
  void operator()(const fs::path &path);
};

}  // namespace core
}  // namespace convolution

#include <convolution/core/Convolver.inl>

#endif  // CONVOLUTION_CORE_CONVOLVER_H