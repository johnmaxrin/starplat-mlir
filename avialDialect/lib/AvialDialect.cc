#include "include/AvialDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/TypeID.h"
#include "mlir/IR/TypeSupport.h"

#include "tblgen/Dialect.cpp.inc"

using namespace mlir;
using namespace mlir::avial;



namespace mlir
{
    namespace avial
    {
        void AvialDialect::initialize()
        {
            addTypes<   
                        #define GET_TYPEDEF_CLASS
                        #include "tblgen/Types.cpp.inc"
                    >();
        }
    }


    Type AvialDialect::parseType(DialectAsmParser &parser) const {
 
}

void AvialDialect::printType(mlir::Type type,
                           mlir::DialectAsmPrinter &printer) const {
  
}

}
