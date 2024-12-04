#ifndef AVIALTDFILE
#define AVIALTDFILE

include "mlir/IR/DialectBase.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributes.td"



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


def AvialGraphType : AvialDialectType<"Graph", "graph"> {
  let summary = "A graph type";
}

def AvialNodeType : AvialDialectType<"Node", "node"> {
  let summary = "A node type";
}




class AvialGraphOps<string mnemonic> : Op<AvialDialect, mnemonic>
{
  let summary = "Get nodes of a graph type";
  //let arguments = (ins AvialGraphType:$graph);
  //let results = (outs AvialNodeType:$node);
  //let assemblyFormat = "$graph  attr-dict `:` `(` type($graph)  `)` `->` type($node)";
}

def Avial_GetNodes : AvialGraphOps<"nodes">
{
  let summary = "Get all nodes of a graph!";
}


#endif 

