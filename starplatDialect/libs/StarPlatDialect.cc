#include "includes/StarPlatDialect.h"
#include "includes/StarPlatOps.h"
#include "includes/StarPlatTypes.h"

#include "StarPlatDialect.cpp.inc"

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"



#define GET_TYPEDEF_CLASSES
#include "StarPlatTypes.cpp.inc"

#define GET_OP_CLASSES
#include "StarPlatOps.cpp.inc"

using namespace mlir;
using namespace mlir::starplat;

namespace mlir
{
    namespace starplat 
    {

        void StarPlatDialect::initialize()
        {

            addTypes<
                #define GET_TYPEDEF_LIST
                #include "StarPlatTypes.cpp.inc"
                >();  



            addOperations<
                #define GET_OP_LIST
                #include "StarPlatOps.cpp.inc"
                >();  
        }

        
    }


}

