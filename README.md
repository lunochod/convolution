# Welcome to Convolution
This project aims to illustrate how to use matrix-matrix multiplications to convolve color images using kernels.

# Build

## Install Required Packages
```bash
sudo apt-get install build-essential cimg-dev libspdlog-dev libbenchmark-dev cmake libboost-dev \
                     libpng-dev graphviz doxygen
```

## Checkout the Code
```bash
git clone https://github.com/lunochod/convolution
```

## Configure and Build using CMake
```bash
cd convolution
mkdir build
cd build
cmake ../
make -j
```

## Run the Unit Tests
```bash
make check
```

## Create Doxygen Documentation
```bash
make doxygen
```

## Configure CMake to use a different Compiler
For GCC use:
```bash
cmake -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ ../
```

For LLVM use:
```bash
cmake -DCMAKE_C_COMPILER=/usr/bin/clang -DCMAKE_CXX_COMPILER=/usr/bin/clang++ ../
```

For Intel use:
```bash
cmake -DCMAKE_C_COMPILER=icc -DCMAKE_CXX_COMPILER=icpc ../
```
