#include "mlir/Conversion/Passes.h"
#include "mlir/Pass/PassManager.h"

#include "includes/StarPlatDialect.h"



namespace mlir
{
    namespace starplat
    {
#define GEN_PASS_DECL_REACHDEF 
#define GEN_PASS_DECL_VERTEXTOEDGE
#include "tblgen2/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "tblgen2/Passes.h.inc"
    }
}

