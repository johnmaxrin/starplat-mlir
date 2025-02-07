#include "include/AvialDialect.h"

#include "tblgen/StarPlatDialect.cpp.inc"

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"

#include "include/AvialTypes.h"
#include "include/AvialOps.h"


#define GET_TYPEDEF_CLASSES
#include "tblgen/StarPlatTypes.cpp.inc"

#define GET_OP_CLASSES
#include "tblgen/StarPlatOps.cpp.inc"

using namespace mlir;
using namespace mlir::avial;

namespace mlir
{
    namespace avial
    {

        void AvialDialect::initialize()
        {
            addTypes<
                #define GET_TYPEDEF_LIST
                #include "tblgen/StarPlatTypes.cpp.inc"
                >();  

            addOperations<
                #define GET_OP_LIST
                #include "tblgen/StarPlatOps.cpp.inc"
                >();  
        }
    }


}
