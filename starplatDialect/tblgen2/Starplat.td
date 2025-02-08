#ifndef STARPLATDIALECT
#define STARPLATDIALECT

include "mlir/IR/DialectBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributes.td"

def StarPlatDialect : Dialect {
    let summary = "Starplat dialect";
    let name = "starplat";
    let cppNamespace = "mlir::starplat";
    let useDefaultTypePrinterParser = 1;
}

class StarplatBaseOp<string mnemonic> : Op<StarPlatDialect, mnemonic>
{
    let summary = "Starplat base operation class.";
}




def FuncOp : StarplatBaseOp<"func">
{
    let summary = "Defines a function in the Starplat Dialect";
    let description = [{This operation represents a function definiton in the Starplat dialect.
                        It contains a name, other related arguments and an optional return type.
                        It has a single region for the function body.
                        }];

  let arguments = (ins 
                    SymbolNameAttr:$funcName,
                    TypeAttrOf<FunctionType>:$funcType,
                    OptionalAttr<ArrayAttr>:$argNames // Argument Attributes 
                  );

  let regions = (region AnyRegion:$body);



  let extraClassDeclaration = [{
    static FuncOp create(Location location, StringRef name, FunctionType funcType);
    static FuncOp create(Location location, StringRef name, FunctionType funcType, ArrayAttr argNames);
                         }];

}

// TYPES

class StarPlatDialectType<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<StarPlatDialect, name, traits> {
  let mnemonic = typeMnemonic;
}


def GraphType : StarPlatDialectType<"Graph", "graph"> {
  let summary = "Graph type";
}

def NodeType : StarPlatDialectType<"Node", "node"> {
  let summary = "Node type";
}

def PropNodeType : StarPlatDialectType<"PropNode", "propNode">{
  let summary = "Property of Node";
  let parameters = (ins "Type":$parameter);
  let assemblyFormat = "`<` $parameter `>`";

}

def PropEdgeType : StarPlatDialectType<"PropEdge", "propEdge">{
  let summary = "Property of Edge";
  let parameters = (ins "Type":$parameter);
  let assemblyFormat = "`<` $parameter `>`";
}


#endif 