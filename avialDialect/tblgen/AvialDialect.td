include "mlir/IR/DialectBase.td"

def AvialDialect : Dialect {
    let summary = "Contains all the graph related operations in starplat";
    let name = "avial";
    let cppNamespace = "::avial";
}