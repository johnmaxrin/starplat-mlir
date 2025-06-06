cmake_minimum_required(VERSION 3.20.0)
project(Avial)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  find_package(MLIR REQUIRED CONFIG)

  message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

  set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
  set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)
  include(HandleLLVMOptions)
else()
  # Build via external projects mechanism
  set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir)
  set(MLIR_INCLUDE_DIR ${MLIR_MAIN_SRC_DIR}/include)
  set(MLIR_GENERATED_INCLUDE_DIR ${LLVM_BINARY_DIR}/tools/mlir/include)
  set(MLIR_INCLUDE_DIRS "${MLIR_INCLUDE_DIR};${MLIR_GENERATED_INCLUDE_DIR}")
endif()

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)


# add_compile_options(-fsanitize=address)
# add_link_options(-fsanitize=address)

# add_compile_options(-fsanitize=undefined)
# add_link_options(-fsanitize=undefined)

set(NO_RTTI "-fno-rtti")
add_definitions(${NO_RTTI})



# Include parser directory as a subdirectory
add_subdirectory(parser)
add_subdirectory(avialDialect)
add_subdirectory(starplatDialect)

# Include directories for the source files
include_directories(
    ${CMAKE_SOURCE_DIR}/ast       # Include path for AST headers
    ${CMAKE_SOURCE_DIR}/codegen    # Include path for codegen headers
    ${CMAKE_SOURCE_DIR}/avialDialect    
    ${CMAKE_SOURCE_DIR}/starplatDialect    

    ${CMAKE_BINARY_DIR}/parser  # Include path for generated parser files
    ${CMAKE_BINARY_DIR}/avialDialect
    ${CMAKE_BINARY_DIR}/starplatDialect
)

# Add executable and link it with the parser library
add_llvm_executable(app main.cc avialDialect/lib/AvialDialect.cc starplatDialect/libs/StarPlatDialect.cc  starplatDialect/libs/StarPlatTransforms.cc
        DEPENDS 
          MLIRStarPlatIncGen 
          MLIRAvialIncGen )


target_link_libraries(app 
    PRIVATE
    parser_lib 
    ${dialect_libs}
    ${FLEX_LIBRARIES} 
    ${BISON_LIBRARIES} 
    ${extension_libs}
    ${translation_libs}
      MLIRAnalysis
          MLIRIR
          MLIRParser
          MLIRPass
          MLIRTranslateLib
          MLIRSupport
          MLIRSideEffectInterfaces
          MLIRTransforms)
