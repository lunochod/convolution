include_directories(${Convolution_SOURCE_DIR}/src)
include_directories(${Convolution_SOURCE_DIR}/include)

list(APPEND io_SOURCES
  ${Convolution_SOURCE_DIR}/src/convolution/io/Image.cpp
)

add_library(io SHARED ${io_SOURCES} )

add_executable(ImageTest ${Convolution_SOURCE_DIR}/src/convolution/io/tests/ImageTest.cpp)
target_link_libraries(ImageTest io gtest_main -lm -lpthread -lX11)
add_test(io::ImageTest ImageTest)
add_dependencies(check ImageTest)
