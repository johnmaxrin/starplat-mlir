get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)


set(LLVM_TARGET_DEFINITIONS AvialDialect.td)
# mlir_tablegen(Ops.h.inc -gen-op-decls)
# mlir_tablegen(Ops.cpp.inc -gen-op-defs)
# mlir_tablegen(Dialect.h.inc -gen-dialect-decls)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(MLIRAvialIncGen)

# Define the location for generated include files
set(GENERATED_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR})
file(MAKE_DIRECTORY ${GENERATED_INCLUDE_DIR})



