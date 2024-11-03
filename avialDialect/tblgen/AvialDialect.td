#include "mlir/IR/DialectBase.td"

def AvialDialect : Dialect {
    let summary = "Contains all the graph related operations in starplat";
    let name = avialDialect;
    let cppNamespace = "::avialDialect"
}