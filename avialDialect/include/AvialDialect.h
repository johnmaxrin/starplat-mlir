#ifndef MLIRDIALECT
#define MLIRDIALECT

#include "mlir/IR/Dialect.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"

#include "mlir/IR/OpImplementation.h" // For Op implementation utilities

#include "tblgen/Dialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "tblgen/Types.h.inc"

#endif