get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)




set(LLVM_TARGET_DEFINITIONS Starplat.td)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=starplat)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=starplat)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
mlir_tablegen(Types.h.inc -gen-typedef-decls )
mlir_tablegen(Types.cpp.inc -gen-typedef-defs )
  
add_public_tablegen_target(MLIRStarplatIncGen)

# Define the location for generated include files
set(GENERATED_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR})
file(MAKE_DIRECTORY ${GENERATED_INCLUDE_DIR})

