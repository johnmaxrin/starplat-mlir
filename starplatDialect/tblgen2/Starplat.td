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

class StarplatBaseOp<string mnemonic, list<Trait> traits = []> : Op<StarPlatDialect, mnemonic, traits>
{
    let summary = "Starplat base operation class.";
}




def FuncOp : StarplatBaseOp<"func",[IsolatedFromAbove, SymbolTable, Symbol]>
{
    let summary = "Defines a function in the Starplat Dialect";
    let description = [{This operation represents a function definiton in the Starplat dialect.
                        It contains a name, other related arguments and an optional return type.
                        It has a single region for the function body.
                        }];

  let arguments = (ins 
                    SymbolNameAttr:$sym_name,
                    TypeAttrOf<FunctionType>:$funcType,
                    OptionalAttr<ArrayAttr>:$argNames // Argument Attributes 
                  );

  let regions = (region AnyRegion:$body);



  let extraClassDeclaration = [{
    static FuncOp create(Location location, StringRef strVal, int intVal);
                         }];

}

def DeclareOp : StarplatBaseOp<"declare", [Symbol]>
{
  let summary = "Declares a variable of a specified type and returns its SSA value.";
  let description = [{
                          The declare operation is used to define a variable within the Starplat dialect. 
                          The variable is associated with a specific type (e.g., integers, 
                          floating-point numbers, or custom types such as propNode), 
                          and it returns an SSA value representing the declared variable. 
  }];

  let arguments = (ins TypeAttr :$type, SymbolNameAttr:$sym_name);

  let results = (outs AnyType);

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

def ConstOp : StarplatBaseOp<"const", [Symbol]> 
{
  let summary = "Const Operaiton";
  let arguments = (ins
                    TypedAttrInterface:$value, SymbolNameAttr:$sym_name
                  );

  let results = (outs AnyType:$retValue);

    let extraClassDeclaration = [{
      static ConstOp create(Location location, Value result, Value argument);
  }];
}

def AttachNodePropertyOp : StarplatBaseOp<"attachNodeProperty">
{
  let summary = "Attach Node Property. Attach this property with all nodes in the given graph";

  let arguments = (ins 
                     Variadic<AnyType>:$operands
                   );
 }

 def SetNodePropertyOp : StarplatBaseOp<"setNodeProperty">
 {
    let summary = "Set Node property";

    let arguments = ( ins
                      AnyType : $node,
                      AnyType : $property,
                      AnyType : $value
                    );
 }

def GetNodePropertyOp : StarplatBaseOp<"getNodeProperty">
 {
    let summary = "Get Node property";

    let arguments = ( ins
                      AnyType : $node,
                      Builtin_StringAttr : $property
                    );
    let results = ( outs
                    AnyType : $result
    );
 }

def GetEdgePropertyOp : StarplatBaseOp<"getEdgeProperty">
 {
    let summary = "Get Edge property";

    let arguments = ( ins
                      AnyType : $node,
                      Builtin_StringAttr : $property 
                    );
    let results = ( outs
                    AnyType : $result
    );
 }

 def FixedPointUntilOp : StarplatBaseOp<"fixedPointUntil">
 {
    let summary = "Fixed-point operation for iterative graph processing";
    
    let description = [{
    This operation performs a fixed-point iteration over graph nodes and edges
    until a convergence condition is met.
    }];

    let arguments = ( ins Variadic<AnyType> : $argumets,
                          OptionalAttr<ArrayAttr> : $terminationCondition
    );
    
    let regions = (region AnyRegion:$body);

 }

 def ForAllOp : StarplatBaseOp<"forall">
 {
    let summary = "Forall loop";
    let description = [{"For all operation description."}];

    let arguments = ( ins 
                      Variadic<AnyType> : $arguments,
                      OptionalAttr<ArrayAttr> : $loopattributes 
                    );
    
    let regions = (region AnyRegion:$body);
 }

 def GetEdgeOp : StarplatBaseOp<"getedge">
 {
    let summary = "Return Edge between node1 and node 2";

    let arguments = ( ins 
                      AnyType : $graph,
                      AnyType : $node1,
                      AnyType : $node2
    );

    let results = ( outs
                    AnyType : $edge
    );
 }


 def MinOp : StarplatBaseOp<"min">
 {
    let summary = "Min Operation of starplat";

    let arguments = ( ins 
                      AnyType : $lhs1,
                      AnyType : $lhs2,
                      AnyType : $rhs1,
                      AnyType : $rhs2
    );

    let results = ( outs
                    AnyType : $result1 
    );

 }

 def endOp : StarplatBaseOp<"end", [Terminator]>
 {
    let summary = "End of the operation";
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

def NodesType : StarPlatDialectType<"Nodes", "nodes">
{
  let summary = "For More than one node";
}

def EdgeType : StarPlatDialectType<"Edge", "edge">
{
  let summary = "Starplat Edge Type";
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

// End of types



#endif 