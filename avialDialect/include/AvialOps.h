#ifndef AVIALOPS_H
#define AVIALOPS_H

#include "AvialDialect.h"
#include "AvialTypes.h"

#include "mlir/IR/BuiltinOps.h"    // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Dialect.h"       // from @llvm-project

#define GET_OP_CLASSES
#include "tblgen/Ops.h.inc"


#endif  // AVIALOPS_H