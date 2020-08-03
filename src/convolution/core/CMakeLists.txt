include_directories(${Convolution_SOURCE_DIR}/src)
include_directories(${Convolution_SOURCE_DIR}/include)

list(APPEND core_SOURCES
  ${Convolution_SOURCE_DIR}/src/convolution/core/ImageReader.cpp
  ${Convolution_SOURCE_DIR}/src/convolution/core/Convolver.cpp
)

add_library(core SHARED ${core_SOURCES} )

add_executable(MatrixMultiplicationTest ${Convolution_SOURCE_DIR}/src/convolution/core/tests/MatrixMultiplicationTest.cpp)
target_link_libraries(MatrixMultiplicationTest gtest_main)
add_test(core::MatrixMultiplicationTest MatrixMultiplicationTest)
add_dependencies(check MatrixMultiplicationTest)

add_executable(FilterTest ${Convolution_SOURCE_DIR}/src/convolution/core/tests/FilterTest.cpp)
target_link_libraries(FilterTest gtest_main)
add_test(core::FilterTest FilterTest)
add_dependencies(check FilterTest)

add_executable(ImageReaderTest ${Convolution_SOURCE_DIR}/src/convolution/core/tests/ImageReaderTest.cpp)
target_link_libraries(ImageReaderTest core gtest_main -lm -lpthread -lX11)
add_test(core::ImageReaderTest ImageReaderTest)
add_dependencies(check ImageReaderTest)

add_executable(ConvolverTest ${Convolution_SOURCE_DIR}/src/convolution/core/tests/ConvolverTest.cpp)
target_link_libraries(ConvolverTest core gtest_main -lm -lpthread -lX11)
add_test(core::ConvolverTest ConvolverTest)
add_dependencies(check ConvolverTest)
