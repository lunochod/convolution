namespace convolution {
namespace core {

template <uint32_t alignment>
void Convolver<alignment>::operator()(const fs::path &path) {
  if (!io.read(path)) {
    spdlog::error("Image file {} not found.", path.c_str());
    return;
  }
  // we require the column buffer to be in column-major order for core::mult()
  io.img2col<core::MatrixOrder::kColumnMajor, alignment>(*filterPtr);

  uint8_t *colBuffer = io.getColumnBuffer()->data();
  uint8_t *filterBuffer = filterPtr->getColumnBuffer();

  const uint32_t M = io.width() * io.height();
  const uint32_t N = core::getAlignedSize<uint32_t, alignment>(filterPtr->numOutputChannels());
  const uint32_t K = core::getAlignedSize<uint32_t, alignment>(filterPtr->height() * filterPtr->width() * filterPtr->numInputChannels());

  auto outputBuffer = io.getTransformBuffer();
  memset(outputBuffer->data(), 0, outputBuffer->size());

  bool didNotOverflow = core::mult<uint8_t, core::MatrixOrder::kColumnMajor, core::MatrixOrder::kColumnMajor, core::MatrixOrder::kRowMajor, alignment, true>(M, N, K, outputBuffer->data(), colBuffer, filterBuffer);
  //bool didNotOverflow = core::gemm<uint8_t, core::MatrixOrder::kColumnMajor, core::MatrixOrder::kColumnMajor, core::MatrixOrder::kRowMajor, true>(M, N, K, outputBuffer->data(), colBuffer, filterBuffer);
  if (!didNotOverflow) {
    spdlog::critical("Overflow detected in core::mult<uint8_t>");
    throw "Overflow detected in core::mult<uint8_t>";
  }

  core::transpose<uint8_t, core::MatrixOrder::kColumnMajor>(M, N, outputBuffer->data());

  for (uint32_t oc = 0; oc < filterPtr->numOutputChannels(); ++oc) {
    auto filename = std::string(path.stem().c_str()) + "_" + std::to_string(oc) + ".png";
    fs::path oPath = path.parent_path() / filename;
    io.write<alignment>(oPath, oc);
  }
}

}  // namespace core
}  // namespace convolution