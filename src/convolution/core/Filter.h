#ifndef CONVOLUTION_CORE_FILTER_H
#define CONVOLUTION_CORE_FILTER_H

#include <convolution/core/logging.h>
#include <convolution/core/math.h>

#include <cstdint>
#include <memory>
#include <stdexcept>

namespace convolution {
namespace core {

template <typename T>
class IFilter {
 public:
  virtual ~IFilter(){};
  virtual uint32_t height() const = 0;
  virtual uint32_t width() const = 0;
  virtual uint32_t numInputChannels() const = 0;
  virtual uint32_t numOutputChannels() const = 0;

  virtual uint32_t leftPadding() const = 0;
  virtual uint32_t rightPadding() const = 0;
  virtual uint32_t topPadding() const = 0;
  virtual uint32_t bottomPadding() const = 0;

  virtual T *getFilterBuffer() = 0;
  virtual const T *getFilterBuffer() const = 0;

  virtual T *getColumnBuffer() = 0;
  virtual const T *getColumnBuffer() const = 0;
};

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