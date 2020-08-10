namespace convolution {
namespace core {

/// address calculation into the column buffer
template <uint32_t alignment>
uint32_t Convolver<alignment>::calcColumnBufferOffset(const uint32_t ix, const uint32_t iy, const uint32_t ic, const uint32_t fx, const uint32_t fy) const {
  const uint32_t pixelIndex = io.width() * iy + ix;
  const uint32_t filterSize = filterPtr->width() * filterPtr->height();
  const uint32_t columBufferWidthAligned = core::getAlignedSize<uint32_t, alignment>(filterSize * io.channels());
  return pixelIndex * columBufferWidthAligned + ic * filterSize + filterPtr->width() * fy + fx;
}

template <uint32_t alignment>
template <core::MatrixOrder order>
bool Convolver<alignment>::img2col() {
  auto imgBufferPtr = io.getImageBuffer();

  if (!imgBufferPtr) {
    spdlog::error("Image buffer is not initialized, failed to initialize column buffer.");
    return false;
  }

  if (imgBufferPtr->empty()) {
    spdlog::error("Image buffer is empty, failed to initialize column buffer.");
    return false;
  }

  IFilter<uint8_t> &filter = *filterPtr;

  const uint32_t imgWidth = io.width();
  const uint32_t imgHeight = io.height();
  const uint32_t imgChannels = io.channels();
  const uint32_t filterWidth = filter.width();
  const uint32_t filterHeight = filter.height();
  const uint32_t paddingWidth = filter.leftPadding();
  const uint32_t paddingHeight = filter.topPadding();
  const uint32_t columnBufferWidth = filterWidth * filterHeight * io.channels();
  const uint32_t columnBufferHeight = io.pixels();
  const uint32_t columnBufferWidthAligned = core::getAlignedSize<uint32_t, alignment>(columnBufferWidth);

  // create a clear a line buffer with sufficient space for left and right padding
  StorageT lineBuffer(imgWidth + filterWidth - 1);
  memset(lineBuffer.data(), 0, lineBuffer.size());

  // resize and clear the column buffer
  colBufferPtr->resize(columnBufferHeight * columnBufferWidthAligned);
  memset(colBufferPtr->data(), 0, colBufferPtr->size());

  // resize and clear the transform buffer
  transformBufferPtr->resize(io.pixels() * core::getAlignedSize<uint32_t, alignment>(filter.numOutputChannels()));
  memset(transformBufferPtr->data(), 0, transformBufferPtr->size());

  // iterate over each channel
  for (uint32_t img_c = 0; img_c < imgChannels; ++img_c) {
    // iterate over image line-by-line vertically
    for (uint32_t img_y = 0; img_y < imgHeight; ++img_y) {
      // copy current image line into the lineBuffer and add horizontal padding
      uint32_t imgOffset = io.calcImageBufferOffset(0, img_y, img_c);
      memcpy(lineBuffer.data() + paddingWidth, imgBufferPtr->data() + imgOffset, imgWidth);
      auto bgn = lineBuffer.begin();
      // iterate over the current image line pixel-by-pixel horizontally
      for (uint32_t img_x = 0; img_x < imgWidth; ++img_x, ++bgn) {
        // iterate vertical over the filter
        for (uint32_t filter_y = 0; filter_y < filterHeight; ++filter_y) {
          // each filter_y position corresponds to a single copy of the data at position bgn into the column buffer
          // calculate the destination line in the image for the current copy
          const int32_t dst_y = img_y - paddingHeight + filter_y;
          if (dst_y >= 0 && dst_y < (int32_t)imgHeight) {
            auto wIt = (*colBufferPtr).begin() + calcColumnBufferOffset(img_x, dst_y, img_c, 0, filterHeight - filter_y - 1);
            std::copy(bgn, bgn + filterWidth, wIt);
          }
        }
      }
    }
  }

  if constexpr (order == core::MatrixOrder::kColumnMajor) {
    const uint32_t N = columnBufferWidthAligned;
    const uint32_t M = io.pixels();
    core::transpose<uint8_t, core::MatrixOrder::kRowMajor>(M, N, colBufferPtr->data(), transformBufferPtr->data());
  }

  return true;
}

template <uint32_t alignment>
void Convolver<alignment>::operator()(const fs::path &path) {
  if (!io.read(path)) {
    spdlog::error("Image file {} not found.", path.c_str());
    return;
  }
  // we require the column buffer to be in column-major order for core::mult()
  img2col<core::MatrixOrder::kColumnMajor>();

  uint8_t *colBuffer = getColumnBuffer()->data();
  uint8_t *filterBuffer = filterPtr->getColumnBuffer();

  const uint32_t M = io.width() * io.height();
  const uint32_t N = core::getAlignedSize<uint32_t, alignment>(filterPtr->numOutputChannels());
  const uint32_t K = core::getAlignedSize<uint32_t, alignment>(filterPtr->height() * filterPtr->width() * filterPtr->numInputChannels());

  auto output = getTransformBuffer();
  memset(output->data(), 0, output->size());

  bool didNotOverflow = core::mult<uint8_t, core::MatrixOrder::kColumnMajor, core::MatrixOrder::kColumnMajor, core::MatrixOrder::kRowMajor, alignment, true>(M, N, K, output->data(), colBuffer, filterBuffer);
  spdlog::info("{}", __LINE__);
  //bool didNotOverflow = core::gemm<uint8_t, core::MatrixOrder::kColumnMajor, core::MatrixOrder::kColumnMajor, core::MatrixOrder::kRowMajor, true>(M, N, K, outputBuffer->data(), colBuffer, filterBuffer);
  if (!didNotOverflow) {
    spdlog::critical("Overflow detected in core::mult<uint8_t>");
    throw "Overflow detected in core::mult<uint8_t>";
  }

  core::transpose<uint8_t, core::MatrixOrder::kColumnMajor>(M, N, output->data());

  auto addr = [&](const uint32_t img_x, const uint32_t img_y, const uint32_t oc) {
    const uint32_t pixelIndex = io.width() * img_y + img_x;
    return pixelIndex * core::getAlignedSize<uint32_t, alignment>(filterPtr->numOutputChannels()) + oc;
  };

  for (uint32_t oc = 0; oc < filterPtr->numOutputChannels(); ++oc) {
    auto filename = std::string(path.stem().c_str()) + "_" + std::to_string(oc) + ".png";
    fs::path oPath = path.parent_path() / filename;

    auto imageBuffer = io.getImageBuffer();

    for (uint32_t img_y = 0; img_y < io.height(); ++img_y) {
      for (uint32_t img_x = 0; img_x < io.width(); ++img_x) {
        uint32_t read = addr(img_x, img_y, oc);
        uint32_t write = io.calcImageBufferOffset(img_x, img_y, 0);
        (*imageBuffer)[write] = (*transformBufferPtr)[read];
      }
    }

    io.write(oPath, oc);
  }
}

template <uint32_t alignment>
typename Convolver<alignment>::StoragePtr Convolver<alignment>::getColumnBuffer() const {
  return colBufferPtr;
}

template <uint32_t alignment>
typename Convolver<alignment>::StoragePtr Convolver<alignment>::getTransformBuffer() const {
  return transformBufferPtr;
}

}  // namespace core
}  // namespace convolution