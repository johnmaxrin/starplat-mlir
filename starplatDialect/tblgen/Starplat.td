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

  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<FunctionType>:$function_type);

  let regions = (region AnyRegion:$body);

  let builders = [OpBuilder<(ins
    "StringRef":$name)
  >];

}


#endif 