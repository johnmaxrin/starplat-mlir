#include "mlir/IR/Dialect.h"

class AvialDialect : public mlir::Dialect {
public:
  explicit AvialDialect(mlir::MLIRContext *context);

  static llvm::StringRef getDialectNamespace() { return "avialdialect"; }
};
