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
    static FuncOp create(Location location, StringRef strVal, int intVal);
                         }];

}

def DeclareOp : StarplatBaseOp<"declare">
{
  let summary = "Declares a variable of a specified type and returns its SSA value.";
  let description = [{
                          The declare operation is used to define a variable within the Starplat dialect. 
                          The variable is associated with a specific type (e.g., integers, 
                          floating-point numbers, or custom types such as propNode), 
                          and it returns an SSA value representing the declared variable. 
  }];

  let arguments = (ins TypeAttr :$type);
  let results = (outs I32);

  let extraClassDeclaration = [{
      static DeclareOp create(Location location, TypeAttr type);
  }];
}

def AssignmentOp : StarplatBaseOp<"assign">
{
  let summary = "The AssignmentOp assigns a value to a variable";
  let description = [{
                The AssignmentOp is used to perform variable assignments in the Starplat dialect. 
                It operates by taking an SSA value (source) and assigning it to a destination
  }];

  let arguments = (ins AnyType:$lhs, AnyType:$rhs);

  let extraClassDeclaration = [{
      static AssignmentOp create(Location location, Value lhs, Value rhs);
  }];
}

def ConstOp : StarplatBaseOp<"const"> 
{
  let summary = "Const Operaiton";
  let arguments = (ins
                    TypedAttrInterface:$value
                  );

  let results = (outs AnyType:$retValue);

    let extraClassDeclaration = [{
      static ConstOp create(Location location, Value result, Value argument);
  }];
}

// def AttachNodePropertyOp : StarplatBaseOp<"attachNodeProperty">
// {
//   let summary = "Attach Node Property. Attach this property with all nodes in the given graph";

//   let arguments = (ins 
//                     GraphType : $graph,
//                     ArrayAttr : 
//                   );
// }

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