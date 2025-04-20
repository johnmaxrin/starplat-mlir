#include "includes/StarPlatDialect.h"
#include "includes/StarPlatOps.h"
#include "includes/StarPlatTypes.h"

#include "tblgen2/StarPlatDialect.cpp.inc"

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"



#define GET_TYPEDEF_CLASSES
#include "tblgen2/StarPlatTypes.cpp.inc"

#define GET_OP_CLASSES
#include "tblgen2/StarPlatOps.cpp.inc"

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
                #include "tblgen2/StarPlatTypes.cpp.inc"
                >();  



            addOperations<
                #define GET_OP_LIST
                #include "tblgen2/StarPlatOps.cpp.inc"
                >();  
        }



        static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState,::mlir::Type restype, ::mlir::TypeAttr typeAttr, ::mlir::StringAttr sym_name, ::mlir::StringAttr sym_visibility)
        {

        }
    }


}

