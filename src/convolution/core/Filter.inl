#include <cstdint>

namespace convolution {
namespace core {

template <typename T, uint32_t kHeight, uint32_t kWidth, uint32_t kInputChannels, uint32_t kOutputChannels, uint32_t alignment>
Filter<T, kHeight, kWidth, kInputChannels, kOutputChannels, alignment>::Filter() {
  static_assert(kNumElements != 0, "Filter dimensions are ill-defined.");
  static_assert(kWidth % 2 == 1, "Filter width must be odd");
  static_assert(kHeight % 2 == 1, "Filter height must be odd");
  filterBuffer = std::make_shared<StorageT>(kNumElements);
}

template <typename T, uint32_t kHeight, uint32_t kWidth, uint32_t kInputChannels, uint32_t kOutputChannels, uint32_t alignment>
Filter<T, kHeight, kWidth, kInputChannels, kOutputChannels, alignment>::Filter(const StorageT &elements) {
  static_assert(kNumElements != 0, "Filter dimensions are ill-defined.");
  if (kNumElements != elements.size()) {
    spdlog::critical("Filter input data size ({}) doesn't match filter dimensions {}x{}x{}x{}", elements.size(), kHeight, kWidth, kInputChannels, kOutputChannels);
    throw std::out_of_range("Filter input data size doesn't match filter dimensions");
  }
  filterBuffer = std::make_shared<StorageT>(kNumElements);
  memcpy(filterBuffer->data(), elements.data(), elements.size() * sizeof(T));
  colBuffer = std::make_shared<StorageT>(kNumElementsAligned);
  filterToColumn();
}

template <typename T, uint32_t kHeight, uint32_t kWidth, uint32_t kInputChannels, uint32_t kOutputChannels, uint32_t alignment>
Filter<T, kHeight, kWidth, 1, 1, alignment> Filter<T, kHeight, kWidth, kInputChannels, kOutputChannels, alignment>::get(uint32_t icIdx, uint32_t ocIdx) const {
  if (icIdx >= kInputChannels) {
    spdlog::critical("Filter input channel index({}) is out of range [0,{}]", icIdx, kInputChannels - 1);
    throw std::out_of_range("Filter input channel is out of range.");
  }
  if (ocIdx >= kOutputChannels) {
    spdlog::critical("Filter output channel index({}) is out of range [0,{}]", ocIdx, kOutputChannels - 1);
    throw std::out_of_range("Filter output channel is out of range.");
  }
  Filter<T, kHeight, kWidth, 1, 1, alignment> f;
  const size_t offset = kHeight * kWidth * kInputChannels * ocIdx + kHeight * kWidth * icIdx;
  memcpy(f.getFilterBuffer(), filterBuffer->data() + offset, kHeight * kWidth * sizeof(T));
  return f;
}

template <typename T, uint32_t kHeight, uint32_t kWidth, uint32_t kInputChannels, uint32_t kOutputChannels, uint32_t alignment>
T Filter<T, kHeight, kWidth, kInputChannels, kOutputChannels, alignment>::at(uint32_t hIdx, uint32_t wIdx, uint32_t icIdx, uint32_t ocIdx) const {
  const size_t idx = kHeight * kWidth * kInputChannels * ocIdx + kHeight * kWidth * icIdx + kHeight * hIdx + wIdx;
  return (*filterBuffer)[idx];
}

template <typename T, uint32_t kHeight, uint32_t kWidth, uint32_t kInputChannels, uint32_t kOutputChannels, uint32_t alignment>
T &Filter<T, kHeight, kWidth, kInputChannels, kOutputChannels, alignment>::at(uint32_t hIdx, uint32_t wIdx, uint32_t icIdx, uint32_t ocIdx) {
  const size_t idx = kHeight * kWidth * kInputChannels * ocIdx + kHeight * kWidth * icIdx + kHeight * hIdx + wIdx;
  return (*filterBuffer)[idx];
}

template <typename T, uint32_t kHeight, uint32_t kWidth, uint32_t kInputChannels, uint32_t kOutputChannels, uint32_t alignment>
void Filter<T, kHeight, kWidth, kInputChannels, kOutputChannels, alignment>::filterToColumn() {
  // the operation is a transpose on a non-square matrix,
  // for simplicity we use element-wise operation using address lookup
  for (uint32_t oc = 0; oc < kOutputChannels; ++oc) {
    for (uint32_t ic = 0; ic < kInputChannels; ++ic) {
      for (uint32_t fy = 0; fy < kHeight; ++fy) {
        for (uint32_t fx = 0; fx < kWidth; ++fx) {
          uint32_t read = calcFilterBufferOffset(fx, fy, ic, oc);
          uint32_t write = calcColumnBufferOffset(fx, fy, ic, oc);
          (*colBuffer)[write] = (*filterBuffer)[read];
        }
      }
    }
  }
}

template <typename T, uint32_t kHeight, uint32_t kWidth, uint32_t kInputChannels, uint32_t kOutputChannels, uint32_t alignment>
uint32_t Filter<T, kHeight, kWidth, kInputChannels, kOutputChannels, alignment>::calcFilterBufferOffset(const uint32_t fx, const uint32_t fy, const uint32_t ic, uint32_t oc) const {
  return oc * kHeight * kWidth * kInputChannels + ic * kHeight * kWidth + fy * kWidth + fx;
}

template <typename T, uint32_t kHeight, uint32_t kWidth, uint32_t kInputChannels, uint32_t kOutputChannels, uint32_t alignment>
uint32_t Filter<T, kHeight, kWidth, kInputChannels, kOutputChannels, alignment>::calcColumnBufferOffset(const uint32_t fx, const uint32_t fy, const uint32_t ic, uint32_t oc) const {
  const uint32_t vertical = ic * kHeight * kWidth + fy * kWidth + fx;
  const uint32_t horizontal = oc;
  return vertical * core::getAlignedSize<uint32_t, alignment>(kOutputChannels) + horizontal;
}

}  // namespace core
}  // namespace convolution
