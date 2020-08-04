#include <convolution/core/Convolver.h>
#include <convolution/core/MatrixMultiplication.h>

namespace convolution {
namespace core {

template <uint32_t P>
void Convolver<P>::operator()(const fs::path &path) {
  if (!io.read(path)) {
    spdlog::error("Image file {} not found.", path.c_str());
    return;
  }
  // we require the column buffer to be in column-major order for core::mult()
  io.img2col<core::MatrixOrder::kColumnMajor>(*filterPtr);

  uint8_t *colBuffer = io.getColumnBuffer()->data();
  uint8_t *filterBuffer = filterPtr->getColumnBuffer();

  const uint32_t M = io.width() * io.height();
  const uint32_t N = filterPtr->numOutputChannels();
  const uint32_t K = filterPtr->height() * filterPtr->width() * filterPtr->numInputChannels();

  auto outputBuffer = io.getTransformBuffer();
  memset(outputBuffer->data(), 0, M * N);

  bool didNotOverflow = core::mult<uint8_t, core::MatrixOrder::kRowMajor, core::MatrixOrder::kColumnMajor, core::MatrixOrder::kRowMajor, P, true>(M, N, K, outputBuffer->data(), colBuffer, filterBuffer);
  if (!didNotOverflow) {
    spdlog::critical("Overflow detected in core::mult<uint8_t>");
    throw "Overflow detected in core::mult<uint8_t>";
  }

  for (uint32_t oc = 0; oc < filterPtr->numOutputChannels(); ++oc) {
    auto filename = std::string(path.stem().c_str()) + "_" + std::to_string(oc) + ".png";
    fs::path oPath = path.parent_path() / filename;
    io.write(oPath, oc);
  }
}

// instantiate Convolver for specific hardware multiplier size
template class Convolver<8>;

}  // namespace core
}  // namespace convolution