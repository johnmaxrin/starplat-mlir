#ifndef STARPLATOPS_H
#define STARPLATOPS_H

#include "StarPlatDialect.h"

#include "mlir/IR/BuiltinOps.h"    // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Dialect.h"       // from @llvm-project

#define GET_OP_CLASSES
#include "tblgen2/StarPlatOps.h.inc"


#endif  // STARPLATOPS~_H