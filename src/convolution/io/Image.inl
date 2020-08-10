namespace convolution {
namespace io {

/// address calculation into the column buffer
template <uint32_t alignment>
uint32_t Image::calcColumnBufferOffset(const core::IFilter<uint8_t> &filter, const uint32_t ix, const uint32_t iy, const uint32_t ic, const uint32_t fx, const uint32_t fy) const {
  const uint32_t pixelIndex = width() * iy + ix;
  const uint32_t filterSize = filter.width() * filter.height();
  const uint32_t columBufferWidthAligned = core::getAlignedSize<uint32_t, alignment>(filterSize * channels());
  return pixelIndex * columBufferWidthAligned + ic * filterSize + filter.width() * fy + fx;
}

/// \brief convert image to column buffer format
/// The layout of the column buffer format follows: http://15418.courses.cs.cmu.edu/fall2017/lecture/dnn/slide_023
///
/// The number of memory reads and writes required to complete the operation:
///   filterWidth x filterHeight x numChannels x imageWidth x imageHeight
///
/// The innermost loop copies the same data of size filterWidth filterHeight times, this small amount of data
/// is kept in the CPU cache, so that the number of DRAM reads is at most:
///   filterWidth x numChannels x imageWidth x imageHeight
///
/// Since we read the image data line-by-line and iterate pixel-by-pixel the overlapping reads of filterWidth
/// will be cached in the CPU cache, so that the number of DRAM reads is further reduced to:
///   numChannels x imageWidth x imageHeight
///
/// Read Complexity  : O( sizeof(Image) )
/// Write Complexity : O( sizeof(Image) x sizeof(Filter) )
///
template <core::MatrixOrder order, uint32_t alignment>
bool Image::img2col(const core::IFilter<uint8_t> &filter) {
  if (imgBufferPtr->empty()) {
    spdlog::error("Image buffer is empty, failed to initialize column buffer.");
    return false;
  }

  const uint32_t imgWidth = width();
  const uint32_t imgHeight = height();
  const uint32_t imgChannels = channels();
  const uint32_t filterWidth = filter.width();
  const uint32_t filterHeight = filter.height();
  const uint32_t paddingWidth = filter.leftPadding();
  const uint32_t paddingHeight = filter.topPadding();
  const uint32_t columnBufferWidth = filterWidth * filterHeight * channels();
  const uint32_t columnBufferHeight = pixels();
  const uint32_t columnBufferWidthAligned = core::getAlignedSize<uint32_t, alignment>(columnBufferWidth);

  // create a clear a line buffer with sufficient space for left and right padding
  StorageT lineBuffer(imgWidth + filterWidth - 1);
  memset(lineBuffer.data(), 0, lineBuffer.size());

  // resize and clear the column buffer
  colBufferPtr.reset();
  colBufferPtr = std::make_shared<StorageT>(columnBufferHeight * columnBufferWidthAligned);
  StorageT &colBuffer = *colBufferPtr;
  memset(colBuffer.data(), 0, colBuffer.size());

  // resize and clear the transform buffer
  transformBufferPtr.reset();
  transformBufferPtr = std::make_shared<StorageT>(pixels() * core::getAlignedSize<uint32_t, alignment>(filter.numOutputChannels()));
  StorageT &transformBuffer = *transformBufferPtr;
  memset(transformBuffer.data(), 0, transformBuffer.size());

  // iterate over each channel
  for (uint32_t img_c = 0; img_c < imgChannels; ++img_c) {
    // iterate over image line-by-line vertically
    for (uint32_t img_y = 0; img_y < imgHeight; ++img_y) {
      // copy current image line into the lineBuffer and add horizontal padding
      uint32_t imgOffset = calcImageBufferOffset(0, img_y, img_c);
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
            auto wIt = colBuffer.begin() + calcColumnBufferOffset<alignment>(filter, img_x, dst_y, img_c, 0, filterHeight - filter_y - 1);
            std::copy(bgn, bgn + filterWidth, wIt);
          }
        }
      }
    }
  }

  if constexpr (order == core::MatrixOrder::kColumnMajor) {
    const uint32_t N = columnBufferWidthAligned;
    const uint32_t M = pixels();
    core::transpose<uint8_t, core::MatrixOrder::kRowMajor>(M, N, colBuffer.data(), transformBuffer.data());
  }

  return true;
}

/// \brief write transform buffer image to the path provided
template <uint32_t alignment>
bool Image::write(const fs::path &path, const uint32_t oc) const {
  CImg<uint8_t> image(width(), height(), 1, 1);

  const uint32_t numOutputChannels = transformBufferPtr->size() / (width() * height());

  auto addr = [&](const uint32_t img_x, const uint32_t img_y, const uint32_t oc) {
    const uint32_t pixelIndex = width() * img_y + img_x;
    return pixelIndex * core::getAlignedSize<uint32_t, alignment>(numOutputChannels) + oc;
  };

  for (uint32_t img_y = 0; img_y < height(); ++img_y) {
    for (uint32_t img_x = 0; img_x < width(); ++img_x) {
      uint32_t read = addr(img_x, img_y, oc);
      image(img_x, img_y, 0, 0) = (*transformBufferPtr)[read];
    }
  }

  image.save(path.c_str());
  spdlog::info("Write image {} {}x{}x{} {} Byte", path.c_str(), width(), height(), 1, transformBufferPtr->size() / numOutputChannels);
  return true;
}

}  // namespace io
}  // namespace convolution