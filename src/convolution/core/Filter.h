#ifndef CONVOLUTION_CORE_FILTER_H
#define CONVOLUTION_CORE_FILTER_H

#include <convolution/core/logging.h>
#include <convolution/core/math.h>

#include <cstdint>
#include <memory>
#include <stdexcept>

namespace convolution {
namespace core {

/// \brief Abstract Filter interface
/// \tparam T(typename) the C++ type used for data elements of the filter
template <typename T>
class IFilter {
 public:
  virtual ~IFilter(){};
  virtual uint32_t height() const = 0;             ///< returns the filter height in pixels
  virtual uint32_t width() const = 0;              ///< returns filter width in pixels
  virtual uint32_t numInputChannels() const = 0;   ///< returns the number of input channels of the filter
  virtual uint32_t numOutputChannels() const = 0;  ///< returns the number of output channels of the filter

  virtual uint32_t leftPadding() const = 0;    ///< returns the padding required on the left of the image
  virtual uint32_t rightPadding() const = 0;   ///< returns the padding required on the right of the image
  virtual uint32_t topPadding() const = 0;     ///< returns the padding required on the top of the image
  virtual uint32_t bottomPadding() const = 0;  ///< returns the padding required on the bottom of the image

  virtual T *getFilterBuffer() = 0;  ///< returns raw pointer to the filter buffer
  virtual const T *getFilterBuffer() const = 0;

  virtual T *getColumnBuffer() = 0;  ///< returns a raw pointer to the filter in column buffer format
  virtual const T *getColumnBuffer() const = 0;
};

/// \class Filter
/// \brief Implements a 4D filter that can be used for convolution of images
///
///  The Filter class is used to construct a KxN column buffer matrix that can be used for convolution, where:
///    K = kHeight * kWidth * kInputChannels
///    N = kOutputChanels
///  where K and N can be padded to be aligned to the alignment parameter specified
///
/// \see http://15418.courses.cs.cmu.edu/fall2017/lecture/dnn/slide_023
/// \tparam T(typename) the data type used for the elements of the filter
/// \tparam kHeight(uint32_t) filter height
/// \tparam kWidth(uint32_t) filter width
/// \tparam kInputChannels(uint32_t) number of input channels of the filter
/// \tparam kOutputChannels(uint32_t) number of output channels of the filter
/// \tparam alignment(uint32_t) allows to force alignment of the column buffer
template <typename T, uint32_t kHeight, uint32_t kWidth, uint32_t kInputChannels = 1, uint32_t kOutputChannels = 1, uint32_t alignment = 1>
class Filter : public IFilter<T> {
 public:
  using StorageT = std::vector<T>;
  using StoragePtr = std::shared_ptr<StorageT>;
  static constexpr uint32_t kNumElements = kHeight * kWidth * kInputChannels * kOutputChannels;
  static constexpr uint32_t kNumElementsAligned = core::getAlignedSize<uint32_t, alignment>(kHeight * kWidth * kInputChannels) * core::getAlignedSize<uint32_t, alignment>(kOutputChannels);

 private:
  StoragePtr filterBuffer = nullptr;  ///< the input filter buffer
  StoragePtr colBuffer = nullptr;     ///< the column buffer

 protected:
  void filterToColumn();

 public:
  Filter();
  Filter(const StorageT &elements);
  Filter(const Filter &rhs) = delete;
  Filter &operator=(const Filter &rhs) = delete;
  Filter(Filter &&rhs) = default;
  Filter &operator=(Filter &&rhs) = default;
  ~Filter() = default;

  const T *getFilterBuffer() const { return filterBuffer->data(); }
  T *getFilterBuffer() { return filterBuffer->data(); }

  const T *getColumnBuffer() const { return colBuffer->data(); }
  T *getColumnBuffer() { return colBuffer->data(); }

  Filter<T, kHeight, kWidth, 1, 1, alignment> get(uint32_t icIdx, uint32_t ocIdx) const;
  T at(uint32_t hIdx, uint32_t wIdx, uint32_t icIdx, uint32_t ocIdx) const;
  T &at(uint32_t hIdx, uint32_t wIdx, uint32_t icIdx, uint32_t ocIdx);

  virtual uint32_t height() const override { return kHeight; }
  virtual uint32_t width() const override { return kWidth; }
  virtual uint32_t numInputChannels() const override { return kInputChannels; }
  virtual uint32_t numOutputChannels() const override { return kOutputChannels; }

  virtual uint32_t leftPadding() const override { return (kWidth - 1) / 2; }
  virtual uint32_t rightPadding() const override { return (kWidth - 1) / 2; }
  virtual uint32_t topPadding() const override { return (kHeight - 1) / 2; }
  virtual uint32_t bottomPadding() const override { return (kHeight - 1) / 2; }

  /// address calculation into the filter buffer
  uint32_t calcFilterBufferOffset(const uint32_t fx, const uint32_t fy, const uint32_t ic, uint32_t oc) const;

  /// address calculation into the column buffer
  uint32_t calcColumnBufferOffset(const uint32_t fx, const uint32_t fy, const uint32_t ic, uint32_t oc) const;
};

}  // namespace core
}  // namespace convolution

#include <convolution/core/Filter.inl>

#endif