#include "includes/StarPlatDialect.h"

#include "Dialect.cpp.inc"

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"

#include "includes/StarPlatOps.h"




#define GET_OP_CLASSES
#include "Ops.cpp.inc"

using namespace mlir;
using namespace mlir::starplat;

namespace mlir
{
    namespace starplat 
    {

        void StarPlatDialect::initialize()
        {

            addOperations<
                #define GET_OP_LIST
                #include "Ops.cpp.inc"
                >();  
        }
    }


}
