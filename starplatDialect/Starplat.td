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

class StarplatBaseOp<string mnemonic> : Op<StarplatDialect, mnemonic>
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
                        Variadic<AnyType>:$inputs
                    );

    let results = (outs);


}


#endif 