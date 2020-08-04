#ifndef CONVOLUTION_CORE_CONVOLVER_H
#define CONVOLUTION_CORE_CONVOLVER_H

#include <convolution/core/Filter.h>
#include <convolution/core/logging.h>
#include <convolution/io/Image.h>

#include <vector>

namespace convolution {
namespace core {

class Convolver {
  std::shared_ptr<IFilter<uint8_t>> filterPtr;
  io::Image io;

 public:
  explicit Convolver(std::shared_ptr<IFilter<uint8_t>> f) : filterPtr(f), io() {}
  void operator()(const fs::path &path);
};

}  // namespace core
}  // namespace convolution

#endif  // CONVOLUTION_CORE_CONVOLVER_H