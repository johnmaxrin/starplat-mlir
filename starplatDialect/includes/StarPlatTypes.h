#ifndef STARPLATDIALECT_TYPES_H
#define STARPLATDIALECT_TYPES_H

// Required because the .h.inc file refers to MLIR classes and does not itself
// have any includes.
#include "mlir/IR/DialectImplementation.h"

#define GET_TYPEDEF_CLASSES
#include "tblgen2/StarPlatTypes.h.inc"

#endif  // STARPLATDIALECT_TYPES_H