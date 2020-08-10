namespace convolution {
namespace core {

/// \brief calculate an offset into the column buffer
/// \param img_x(const uint32_t) pixel position along the horizontal width of the image in row-major format
/// \param img_y(const uint32_t) pixel position along the vertical height of the image in row-major format
/// \param img_c(const uint32_t) pixel channel
/// \tparam filter_x(const uint32_t) filter position along the horizontal width of the filter in row-major format
/// \tparam filter_y(const uint32_t) filter position along the vertical height of the filter in row-major format
/// \return the offset into the column buffer
template <uint32_t alignment>
uint32_t Convolver<alignment>::calcColumnBufferOffset(const uint32_t img_x, const uint32_t img_y, const uint32_t img_c, const uint32_t filter_x, const uint32_t filter_y) const {
  const uint32_t pixelIndex = img.width() * img_y + img_x;
  const uint32_t filterSize = filterPtr->width() * filterPtr->height();
  const uint32_t columBufferWidthAligned = core::getAlignedSize<uint32_t, alignment>(filterSize * img.channels());
  return pixelIndex * columBufferWidthAligned + img_c * filterSize + filterPtr->width() * filter_y + filter_x;
}

/// \brief convert a multi-channel image into column buffer format suitable to support convolution
/// \tparam order(core::MatrixOrder) the matrix order to be used by the column buffer in support of the matrix-matrix multiplication
/// \return bool true on success, false otherwise
template <uint32_t alignment>
template <core::MatrixOrder order>
bool Convolver<alignment>::img2col() {
  auto imgBufferPtr = img.getImageBuffer();

  if (!imgBufferPtr) {
    spdlog::error("Image buffer is not initialized, failed to initialize column buffer.");
    return false;
  }

  if (imgBufferPtr->empty()) {
    spdlog::error("Image buffer is empty, failed to initialize column buffer.");
    return false;
  }

  IFilter<ColumnDataT> &filter = *filterPtr;

  const uint32_t imgWidth = img.width();
  const uint32_t imgHeight = img.height();
  const uint32_t imgChannels = img.channels();
  const uint32_t filterWidth = filter.width();
  const uint32_t filterHeight = filter.height();
  const uint32_t paddingWidth = filter.leftPadding();
  const uint32_t paddingHeight = filter.topPadding();
  const uint32_t columnBufferWidth = filterWidth * filterHeight * img.channels();
  const uint32_t columnBufferHeight = img.pixels();
  const uint32_t columnBufferWidthAligned = core::getAlignedSize<uint32_t, alignment>(columnBufferWidth);

  // create and clear a line buffer with sufficient space for left and right padding
  std::vector<ColumnDataT> lineBuffer(imgWidth + filterWidth - 1);
  std::fill(lineBuffer.begin(), lineBuffer.end(), 0);

  // resize and clear the column buffer
  colBufferPtr->resize(columnBufferHeight * columnBufferWidthAligned);
  std::fill(colBufferPtr->begin(), colBufferPtr->end(), 0);

  // resize and clear the transform buffer
  transformBufferPtr->resize(img.pixels() * core::getAlignedSize<uint32_t, alignment>(filter.numOutputChannels()));
  std::fill(transformBufferPtr->begin(), transformBufferPtr->end(), 0);

  // iterate over each channel
  for (uint32_t img_c = 0; img_c < imgChannels; ++img_c) {
    // iterate over image line-by-line vertically
    for (uint32_t img_y = 0; img_y < imgHeight; ++img_y) {
      // copy current image line into the lineBuffer and add horizontal padding
      uint32_t imgOffset = img.calcImageBufferOffset(0, img_y, img_c);
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

  // in case kColumnMajor format is requested we need to transpose the column buffer
  if constexpr (order == core::MatrixOrder::kColumnMajor) {
    const uint32_t N = columnBufferWidthAligned;
    const uint32_t M = img.pixels();
    core::transpose<ColumnDataT, core::MatrixOrder::kRowMajor>(M, N, colBufferPtr->data());
  }

  return true;
}

/// \brief operator to execute the convolution using the image provided at path
/// \param path (const fs:path &) image location on disk
template <uint32_t alignment>
void Convolver<alignment>::operator()(const fs::path &path) {
  if (!img.read(path)) {
    spdlog::error("Image file {} not found.", path.c_str());
    return;
  }

  // transform the image data into column buffer format using column-major order in support of core::mult()
  img2col<core::MatrixOrder::kColumnMajor>();

  ColumnDataT *colBuffer = getColumnBuffer()->data();
  ColumnDataT *filterBuffer = filterPtr->getColumnBuffer();

  const uint32_t M = img.width() * img.height();
  const uint32_t N = core::getAlignedSize<uint32_t, alignment>(filterPtr->numOutputChannels());
  const uint32_t K = core::getAlignedSize<uint32_t, alignment>(filterPtr->height() * filterPtr->width() * filterPtr->numInputChannels());

  auto output = getTransformBuffer();
  std::fill(output->begin(), output->end(), 0);

  bool didNotOverflow = core::mult<TransformDataT, ColumnDataT, core::MatrixOrder::kColumnMajor, core::MatrixOrder::kColumnMajor, core::MatrixOrder::kRowMajor, alignment, true>(M, N, K, output->data(), colBuffer, filterBuffer);
  if (!didNotOverflow) {
    spdlog::critical("Overflow detected in core::mult");
    throw "Overflow detected in core::mult";
  }

  core::transpose<TransformDataT, core::MatrixOrder::kColumnMajor>(M, N, output->data());

  // lambda for address calculation into the output buffer
  auto addr = [&](const uint32_t img_x, const uint32_t img_y, const uint32_t oc) {
    const uint32_t pixelIndex = img.width() * img_y + img_x;
    return pixelIndex * core::getAlignedSize<uint32_t, alignment>(filterPtr->numOutputChannels()) + oc;
  };

  // write an 8Bit image for each output channel of the filter
  for (uint32_t oc = 0; oc < filterPtr->numOutputChannels(); ++oc) {
    auto filename = std::string(path.stem().c_str()) + "_" + std::to_string(oc) + ".png";
    fs::path oPath = path.parent_path() / filename;

    auto imageBuffer = img.getImageBuffer();

    for (uint32_t img_y = 0; img_y < img.height(); ++img_y) {
      for (uint32_t img_x = 0; img_x < img.width(); ++img_x) {
        uint32_t read = addr(img_x, img_y, oc);
        uint32_t write = img.calcImageBufferOffset(img_x, img_y, 0);
        (*imageBuffer)[write] = (*transformBufferPtr)[read];
      }
    }

    img.write(oPath, oc);
  }
}

template <uint32_t alignment>
typename Convolver<alignment>::ColumnBufferPtr Convolver<alignment>::getColumnBuffer() const {
  return colBufferPtr;
}

template <uint32_t alignment>
typename Convolver<alignment>::TransformBufferPtr Convolver<alignment>::getTransformBuffer() const {
  return transformBufferPtr;
}

template <uint32_t alignment>
bool Convolver<alignment>::read(const fs::path &path) {
  return img.read(path);
}

}  // namespace core
}  // namespace convolution