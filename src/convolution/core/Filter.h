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
  std::shared_ptr<std::vector<T>> mem = nullptr;
  static constexpr uint32_t kNumElements = kHeight * kWidth * kInputChannels * kOutputChannels;

 public:
  Filter() {
    static_assert(kNumElements != 0, "Filter dimensions are ill-defined.");
    static_assert(kWidth % 2 == 1, "Filter with must not be even");
    static_assert(kHeight % 2 == 1, "Filter with must not be even");
    mem = std::make_shared<std::vector<T>>(kHeight * kWidth * kInputChannels * kOutputChannels);
  }

  Filter(const std::vector<T> &elements) {
    static_assert(kNumElements != 0, "Filter dimensions are ill-defined.");
    if (kNumElements != elements.size()) {
      spdlog::critical("Filter input data size ({}) doesn't match Filter dimensions {}x{}x{}x{}", elements.size(), kHeight, kWidth, kInputChannels, kOutputChannels);
      throw std::out_of_range("Filter input data size doesn't match Filter dimensions");
    }
    mem = std::make_shared<std::vector<T>>(kHeight * kWidth * kInputChannels * kOutputChannels);
    memcpy(this->data(), elements.data(), elements.size() * sizeof(T));
  }

  Filter(const Filter &rhs) = delete;
  Filter &operator=(const Filter &rhs) = delete;

  Filter(Filter &&rhs) {
    mem = rhs.mem;
  }
  Filter &operator=(Filter &&rhs) {
    mem = rhs.mem;
    return *this;
  }
  ~Filter() = default;

  const T *data() const { return mem.get()->data(); }
  T *data() { return mem.get()->data(); }

  Filter<T, kHeight, kWidth> get(uint32_t icIdx, uint32_t ocIdx) const {
    if (icIdx >= kInputChannels) {
      spdlog::critical("Filter input channel index({}) is out of range [0,{}]", icIdx, kInputChannels - 1);
      throw std::out_of_range("Filter input channel is out of range.");
    }
    if (ocIdx >= kOutputChannels) {
      spdlog::critical("Filter output channel index({}) is out of range [0,{}]", ocIdx, kOutputChannels - 1);
      throw std::out_of_range("Filter output channel is out of range.");
    }
    Filter<T, kHeight, kWidth> f;
    const size_t offset = kHeight * kWidth * kInputChannels * ocIdx + kHeight * kWidth * icIdx;
    memcpy(f.data(), this->data() + offset, kHeight * kWidth * sizeof(T));
    return f;
  }

  T at(uint32_t hIdx, uint32_t wIdx, uint32_t icIdx, uint32_t ocIdx) const {
    const size_t idx = kHeight * kWidth * kInputChannels * ocIdx + kHeight * kWidth * icIdx + kHeight * hIdx + wIdx;
    return data()[idx];
  }

  T &at(uint32_t hIdx, uint32_t wIdx, uint32_t icIdx, uint32_t ocIdx) {
    const size_t idx = kHeight * kWidth * kInputChannels * ocIdx + kHeight * kWidth * icIdx + kHeight * hIdx + wIdx;
    return data()[idx];
  }

  virtual uint32_t height() const override {
    return kHeight;
  }

  virtual uint32_t width() const override {
    return kWidth;
  }

  virtual uint32_t leftPadding() const override {
    return (kWidth - 1) / 2;
  }

  virtual uint32_t rightPadding() const override {
    return (kWidth - 1) / 2;
  }

  virtual uint32_t topPadding() const override {
    return (kHeight - 1) / 2;
  }

  virtual uint32_t bottomPadding() const override {
    return (kHeight - 1) / 2;
  }
};

}  // namespace core
}  // namespace convolution
#endif