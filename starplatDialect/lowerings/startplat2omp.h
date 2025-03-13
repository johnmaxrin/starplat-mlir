#include "mlir/Conversion/Passes.h"
#include "mlir/Pass/PassManager.h"

#include "includes/StarPlatDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"

namespace mlir
{
    namespace starplat
    {
        #define GEN_PASS_DEF_CONVERTSTARTPLATIRTOOMPPASS
        #include "tblgen2/Passes.h.inc"

        struct ConvertStartPlatIRToOMPPass : public mlir::starplat::impl::ConvertStartPlatIRToOMPPassBase<ConvertStartPlatIRToOMPPass>
        {
            using ConvertStartPlatIRToOMPPassBase::ConvertStartPlatIRToOMPPassBase;

            void runOnOperation() override
            {
                auto mod = getOperation();
                mod->walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *op)
                {
                    op->getContext()->getOrLoadDialect<mlir::LLVM::LLVMDialect>();
                    if(llvm::isa<mlir::starplat::DeclareOp>(op))
                    {
                        // I think we have to learn about pointers, memalloc and all. 
                        // Bye for now. 
                    }
                });
            }
        };
    }
}
