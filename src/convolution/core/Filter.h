#ifndef CONVOLUTION_CORE_FILTER_H
#define CONVOLUTION_CORE_FILTER_H

#include <convolution/core/logging.h>

#include <cstdint>
#include <memory>
#include <stdexcept>

namespace convolution {
namespace core {

class IFilter {
 public:
  virtual ~IFilter(){};
  virtual uint32_t height() const = 0;
  virtual uint32_t width() const = 0;
  virtual uint32_t leftPadding() const = 0;
  virtual uint32_t rightPadding() const = 0;
  virtual uint32_t topPadding() const = 0;
  virtual uint32_t bottomPadding() const = 0;
};

template <typename T, uint32_t kHeight, uint32_t kWidth, uint32_t kInputChannels = 1, uint32_t kOutputChannels = 1>
class Filter : public IFilter {
 public:
  using StorageT = std::vector<T>;
  using StoragePtr = std::shared_ptr<StorageT>;
  static constexpr uint32_t kNumElements = kHeight * kWidth * kInputChannels * kOutputChannels;

 private:
  StoragePtr filterBuffer = nullptr;  ///< the input filter buffer
  StoragePtr colBuffer = nullptr;     ///< the column buffer

 public:
  Filter();
  Filter(const StorageT &elements);
  Filter(const Filter &rhs) = delete;
  Filter &operator=(const Filter &rhs) = delete;
  Filter(Filter &&rhs) = default;
  Filter &operator=(Filter &&rhs) = default;
  ~Filter() = default;

  const T *data() const { return filterBuffer.get()->data(); }
  T *data() { return filterBuffer.get()->data(); }

  Filter<T, kHeight, kWidth> get(uint32_t icIdx, uint32_t ocIdx) const;
  T at(uint32_t hIdx, uint32_t wIdx, uint32_t icIdx, uint32_t ocIdx) const;
  T &at(uint32_t hIdx, uint32_t wIdx, uint32_t icIdx, uint32_t ocIdx);

  virtual uint32_t height() const override { return kHeight; }
  virtual uint32_t width() const override { return kWidth; }

  virtual uint32_t leftPadding() const override { return (kWidth - 1) / 2; }
  virtual uint32_t rightPadding() const override { return (kWidth - 1) / 2; }
  virtual uint32_t topPadding() const override { return (kHeight - 1) / 2; }
  virtual uint32_t bottomPadding() const override { return (kHeight - 1) / 2; }

  void filterToColumn();
};

}  // namespace core
}  // namespace convolution

#include <convolution/core/Filter.inl>

#endif