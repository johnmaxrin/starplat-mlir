#ifndef AVIALTYPES_H
#define AVIALTYPES_H

// Required because the .h.inc file refers to MLIR classes and does not itself
// have any includes.
#include "mlir/IR/DialectImplementation.h"

#define GET_TYPEDEF_CLASSES
#include "tblgen/Types.h.inc"

#endif  // AVIALTYPES_H