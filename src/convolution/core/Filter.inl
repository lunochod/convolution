#include <cstdint>

namespace convolution {
namespace core {

template <typename T, uint32_t kHeight, uint32_t kWidth, uint32_t kInputChannels, uint32_t kOutputChannels>
Filter<T, kHeight, kWidth, kInputChannels, kOutputChannels>::Filter() {
  static_assert(kNumElements != 0, "Filter dimensions are ill-defined.");
  static_assert(kWidth % 2 == 1, "Filter width must be odd");
  static_assert(kHeight % 2 == 1, "Filter width must be odd");
  filterBuffer = std::make_shared<StorageT>(kNumElements);
}

template <typename T, uint32_t kHeight, uint32_t kWidth, uint32_t kInputChannels, uint32_t kOutputChannels>
Filter<T, kHeight, kWidth, kInputChannels, kOutputChannels>::Filter(const StorageT &elements) {
  static_assert(kNumElements != 0, "Filter dimensions are ill-defined.");
  if (kNumElements != elements.size()) {
    spdlog::critical("Filter input data size ({}) doesn't match filter dimensions {}x{}x{}x{}", elements.size(), kHeight, kWidth, kInputChannels, kOutputChannels);
    throw std::out_of_range("Filter input data size doesn't match filter dimensions");
  }
  filterBuffer = std::make_shared<StorageT>(kNumElements);
  memcpy(this->data(), elements.data(), elements.size() * sizeof(T));
  colBuffer = std::make_shared<StorageT>();
}

template <typename T, uint32_t kHeight, uint32_t kWidth, uint32_t kInputChannels, uint32_t kOutputChannels>
Filter<T, kHeight, kWidth> Filter<T, kHeight, kWidth, kInputChannels, kOutputChannels>::get(uint32_t icIdx, uint32_t ocIdx) const {
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

template <typename T, uint32_t kHeight, uint32_t kWidth, uint32_t kInputChannels, uint32_t kOutputChannels>
T Filter<T, kHeight, kWidth, kInputChannels, kOutputChannels>::at(uint32_t hIdx, uint32_t wIdx, uint32_t icIdx, uint32_t ocIdx) const {
  const size_t idx = kHeight * kWidth * kInputChannels * ocIdx + kHeight * kWidth * icIdx + kHeight * hIdx + wIdx;
  return data()[idx];
}

template <typename T, uint32_t kHeight, uint32_t kWidth, uint32_t kInputChannels, uint32_t kOutputChannels>
T &Filter<T, kHeight, kWidth, kInputChannels, kOutputChannels>::at(uint32_t hIdx, uint32_t wIdx, uint32_t icIdx, uint32_t ocIdx) {
  const size_t idx = kHeight * kWidth * kInputChannels * ocIdx + kHeight * kWidth * icIdx + kHeight * hIdx + wIdx;
  return data()[idx];
}

}  // namespace core
}  // namespace convolution
