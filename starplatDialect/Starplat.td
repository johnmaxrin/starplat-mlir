#ifndef STARPLATDIALECT
#define STARPLATDIALECT

include "mlir/IR/DialectBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributes.td"

def StarplatDialect : Dialect {
    let summary = "Starplat dialect";
    let name = "starplat";
    let cppNamespace = "mlir::avial";
    let useDefaultTypePrinterParser = 1;
}


#endif 