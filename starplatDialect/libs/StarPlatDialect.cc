#include "includes/StarPlatDialect.h"
#include "includes/StarPlatOps.h"

#include "tblgen2/StarPlatDialect.cpp.inc"

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"





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

            addOperations<
                #define GET_OP_LIST
                #include "tblgen2/StarPlatOps.cpp.inc"
                >();  
        }
    }


}

mlir::Type mlir::starplat::StarPlatDialect::parseType(mlir::DialectAsmParser &parser) const {
  
}

void mlir::starplat::StarPlatDialect::printType(mlir::Type type, mlir::DialectAsmPrinter &printer) const {
//   if (type.isa<PropNodeType>()) {
//     printer << "propNode";
//     return;
//   }

//   printer << "<unknown StarPlat type>";
}

