#include "mlir/Conversion/Passes.h"
#include "mlir/Pass/PassManager.h"

#include "includes/StarPlatDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

// mlir::Operation *generateGraphStruct(mlir::MLIRContext *context)
// {
//     auto structType = LLVM::LLVMStructType::getIdentified(context, "Graph");

//     return;
// }

mlir::LLVM::LLVMStructType createGraphStruct(mlir::IRRewriter *rewriter, mlir::MLIRContext *context);
mlir::LLVM::LLVMStructType createNodeStruct(mlir::IRRewriter *rewriter, mlir::MLIRContext *context);
void lowerFunctionOp(mlir::Operation *op, mlir::IRRewriter *rewriter);
void lowerFixedPointUntilOp(mlir::Operation *op, mlir::IRRewriter *rewriter);
void lowerForAllOp(mlir::Operation *op, mlir::IRRewriter *rewriter);

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
                mlir::Region &region = mod->getRegion(0);
                mlir::Block &block = region.front();

                mlir::IRRewriter rewriter(mod->getContext());
                mlir::OpBuilder::InsertionGuard guard(rewriter);

                auto graphType = createGraphStruct(&rewriter, rewriter.getContext());
                auto nodeType = createNodeStruct(&rewriter, rewriter.getContext());

                for (mlir::Operation &op : block.getOperations())
                {
                    if (llvm::isa<mlir::starplat::FuncOp>(op))
                        lowerFunctionOp(&op, &rewriter);
                }

                // mod->dump();
            }
        };
    }
}

void lowerFunctionOp(mlir::Operation *op, mlir::IRRewriter *rewriter)
{
    mlir::Region &region = op->getRegion(0);
    mlir::Block &block = region.front();
    
    for (mlir::Operation &op : block.getOperations())
    {
        if (llvm::isa<mlir::starplat::FixedPointUntilOp>(op))
            lowerFixedPointUntilOp(&op, rewriter);
    }
}

void lowerFixedPointUntilOp(mlir::Operation *op, mlir::IRRewriter *rewriter)
{
    mlir::Region &region = op->getRegion(0);
    mlir::Block &block = region.front();
    
    for (mlir::Operation &op : block.getOperations())
    {
        if (llvm::isa<mlir::starplat::ForAllOp>(op))
            lowerForAllOp(&op, rewriter);
    }
}

void lowerForAllOp(mlir::Operation *op, mlir::IRRewriter *rewriter)
{
    mlir::Region &region = op->getRegion(0);
    mlir::Block &block = region.front();

    rewriter->setInsertionPointToStart(&block);
    
    for (mlir::Operation &loopOp : block.getOps())
    {
        

       if(llvm::isa<mlir::starplat::ForAllOp>(op))
        {

            auto i32Type = rewriter->getI32Type();
            auto i32O = rewriter->getI32IntegerAttr(0);
            auto i321 = rewriter->getI32IntegerAttr(1);
            auto i3210 = rewriter->getI32IntegerAttr(10);
            auto const0 = rewriter->create<mlir::LLVM::ConstantOp>(rewriter->getUnknownLoc(), i32Type, i32O);
            auto const1 = rewriter->create<mlir::LLVM::ConstantOp>(rewriter->getUnknownLoc(), i32Type, i321);
            auto const10 = rewriter->create<mlir::LLVM::ConstantOp>(rewriter->getUnknownLoc(), i32Type, i3210);
            auto sample = rewriter->create<mlir::scf::ForOp>(rewriter->getUnknownLoc(), const0->getResult(0), const10->getResult(0), const1->getResult(0));

            loopOp.replaceAllUsesWith(sample);
        }
    }
}


mlir::LLVM::LLVMStructType createGraphStruct(mlir::IRRewriter *rewriter, mlir::MLIRContext *context)
{

    auto structType = LLVM::LLVMStructType::getIdentified(context, "Graph");
    // Create Node struct type

    auto ptrType = LLVM::LLVMPointerType::get(context);
    // Define Graph struct body with (Node*, int)
    structType.setBody({ptrType, rewriter->getI32Type(), ptrType}, /*isPacked=*/false);

    // structType.setBody({rewriter->getI32Type()}, false);

    return structType;
}

mlir::LLVM::LLVMStructType createNodeStruct(mlir::IRRewriter *rewriter, mlir::MLIRContext *context)
{
    auto structType = LLVM::LLVMStructType::getIdentified(context, "Node");

    // Create a ptr type
    // auto ptr = mlir::LLVM::LLVMPointerType::get(rewriter->getI32Type());

    structType.setBody({rewriter->getI32Type()}, false);

    return structType;
}
