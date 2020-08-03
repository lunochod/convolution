#ifndef CONVOLUTION_CORE_CONVOLVER_H
#define CONVOLUTION_CORE_CONVOLVER_H

#include <convolution/core/Filter.h>
#include <convolution/core/ImageReader.h>
#include <convolution/core/logging.h>

#include <vector>

namespace convolution {
namespace core {

class Convolver {
  std::shared_ptr<IFilter<uint8_t>> filterPtr;
  ImageReader io;
  std::vector<uint8_t> outputBuffer;

 public:
  explicit Convolver(std::shared_ptr<IFilter<uint8_t>> f) : filterPtr(f), io() {}
  void operator()(const fs::path &path);
};

}  // namespace core
}  // namespace convolution

#endif  // CONVOLUTION_CORE_CONVOLVER_H