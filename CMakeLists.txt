cmake_minimum_required(VERSION 2.9)
project(Avial)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include parser directory as a subdirectory
add_subdirectory(parser)

# Include directories for the source files
include_directories(
    ${CMAKE_SOURCE_DIR}/ast       # Include path for AST headers
    ${CMAKE_SOURCE_DIR}/codegen    # Include path for codegen headers
    ${CMAKE_BINARY_DIR}/parser  # Include path for generated parser files
)

# Add executable and link it with the parser library
add_executable(app main.cc)
target_link_libraries(app parser_lib ${FLEX_LIBRARIES} ${BISON_LIBRARIES})
