#include "mlir/Conversion/Passes.h"
#include "mlir/Pass/PassManager.h"

#include "includes/StarPlatDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

// mlir::Operation *generateGraphStruct(mlir::MLIRContext *context)
// {
//     auto structType = LLVM::LLVMStructType::getIdentified(context, "Graph");

//     return;
// }

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
                mod->walk<mlir::WalkOrder::PreOrder>([&](mlir::starplat::FuncOp funcOp)
                                                     {
                                                         funcOp->getContext()->getOrLoadDialect<mlir::LLVM::LLVMDialect>();
                                                         mlir::IRRewriter rewriter(funcOp->getContext());
                                                         mlir::OpBuilder::InsertionGuard guard(rewriter);

                                                         auto &entryBlock = funcOp.getBody().getBlocks().front();
                                                         rewriter.setInsertionPointToStart(&entryBlock);

                                                         // CSR For Vertex based.
                                                         // COO for Edge based.

                                                         // Check arg ops and declare structs. Eg Graph, Nodes etc
                                                         funcOp->walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *arg)
                                                                                                 {
                            if (auto typeAttr = arg->getAttrOfType<mlir::TypeAttr>("type")) {
                                mlir::Type argType = typeAttr.getValue();
                    
                                // Check if argType is the specific Starplat type
                                if (argType.isa<mlir::starplat::GraphType>()) {
                                    llvm::outs() << "Graph Type!\n";
                                    auto structType = LLVM::LLVMStructType::getIdentified(funcOp->getContext(), "Graph");
                                    structType.setBody({rewriter.getI32Type()},false);

                                    auto alloc = rewriter.create<LLVM::AllocaOp>(rewriter.getUnknownLoc(), LLVM::LLVMPointerType::get(funcOp.getContext()), structType, rewriter.create<LLVM::ConstantOp>(rewriter.getUnknownLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(1)));
                                    
                                }
                            } });
                                                     });
            }
        };
    }
}
