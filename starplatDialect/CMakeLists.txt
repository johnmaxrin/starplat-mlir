get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)




set(LLVM_TARGET_DEFINITIONS Starplat.td)
mlir_tablegen(StarPlatDialect.h.inc -gen-dialect-decls -dialect=starplat)
mlir_tablegen(StarPlatDialect.cpp.inc -gen-dialect-defs -dialect=starplat)
mlir_tablegen(StarPlatOps.h.inc -gen-op-decls)
mlir_tablegen(StarPlatOps.cpp.inc -gen-op-defs)
mlir_tablegen(StarPlatTypes.h.inc -gen-typedef-decls )
mlir_tablegen(StarPlatTypes.cpp.inc -gen-typedef-defs )
mlir_tablegen(Passes.h.inc -gen-pass-decls)
  
add_public_tablegen_target(MLIRStarPlatIncGen)

# Define the location for generated include files
set(GENERATED_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR})
file(MAKE_DIRECTORY ${GENERATED_INCLUDE_DIR})

