include_directories(${Sequoia_SOURCE_DIR}/src)
include_directories(${Sequoia_SOURCE_DIR}/include)

list(APPEND core_SOURCES
  ${Sequoia_SOURCE_DIR}/src/convolution/core/ImageReader.cpp
)

add_library(core SHARED ${core_SOURCES} )

add_executable(MatrixMultiplicationTest ${Sequoia_SOURCE_DIR}/src/convolution/core/tests/MatrixMultiplicationTest.cpp)
target_link_libraries(MatrixMultiplicationTest gtest_main)
add_test(core::MatrixMultiplicationTest MatrixMultiplicationTest)
add_dependencies(check MatrixMultiplicationTest)

add_executable(FilterTest ${Sequoia_SOURCE_DIR}/src/convolution/core/tests/FilterTest.cpp)
target_link_libraries(FilterTest gtest_main)
add_test(core::FilterTest FilterTest)
add_dependencies(check FilterTest)

add_executable(ImageReaderTest ${Sequoia_SOURCE_DIR}/src/convolution/core/tests/ImageReaderTest.cpp)
target_link_libraries(ImageReaderTest core gtest_main -lm -lpthread -lX11)
add_test(core::ImageReaderTest ImageReaderTest)
add_dependencies(check ImageReaderTest)
