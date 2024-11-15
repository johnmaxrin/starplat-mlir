include "mlir/IR/DialectBase.td"
include "mlir/IR/AttrTypeBase.td"

def AvialDialect : Dialect {
    let summary = "Contains all the graph related operations in starplat";
    let name = "avial";
    let cppNamespace = "mlir::avial";
    let useDefaultTypePrinterParser = 1;
}


class AvialDialectType<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<AvialDialect, name, traits> {
  let mnemonic = typeMnemonic;
}

// Here is a simple definition of an "integer" type, with a width parameter.
def AvialGraphType : AvialDialectType<"Graph", "graph"> {
  let summary = "A graph type";
  
}
