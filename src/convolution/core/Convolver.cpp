#include <convolution/core/Convolver.h>
#include <convolution/core/MatrixMultiplication.h>

namespace convolution {
namespace core {

void Convolver::operator()(const fs::path &path) {
  if (!io.read(path)) {
    spdlog::error("Image file {} not found.", path.c_str());
    return;
  }
  io.img2col(*filterPtr);

  uint8_t *colBuffer = io.getColumnBuffer()->data();
  uint8_t *filterBuffer = filterPtr->getColumnBuffer();

  const uint32_t M = io.width() * io.height();
  const uint32_t N = filterPtr->numOutputChannels();
  const uint32_t K = filterPtr->height() * filterPtr->width() * filterPtr->numInputChannels();

  outputBuffer.resize(M * N);
  memset(outputBuffer.data(), 0, M * N);

  bool didNotOverflow = core::gemm<uint8_t, core::MatrixOrder::kRowMajor>(M, N, K, outputBuffer.data(), colBuffer, filterBuffer);
  if (!didNotOverflow) {
    spdlog::critical("Overflow detected in core::gemm<uint8_t>");
    std::abort();
  }
}

}  // namespace core
}  // namespace convolution