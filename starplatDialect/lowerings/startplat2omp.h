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

mlir::LLVM::LLVMStructType createGraphStruct(mlir::IRRewriter *rewriter, mlir::MLIRContext *context);
mlir::LLVM::LLVMStructType createNodeStruct(mlir::IRRewriter *rewriter, mlir::MLIRContext *context);

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
                    llvm::SmallVector<mlir::Operation *, 4> toErase;
                    funcOp->walk<mlir::WalkOrder::PreOrder>([&](mlir::starplat::ArgOp arg)
                    {
                            if (auto typeAttr = arg->getAttrOfType<mlir::TypeAttr>("type")) {
                                mlir::Type argType = typeAttr.getValue();
                    
                                // Check if argType is the specific Starplat type
                                if (argType.isa<mlir::starplat::GraphType>()) {

                                    auto graphStruct = createGraphStruct(&rewriter, funcOp->getContext());
                                    auto alloc = rewriter.create<LLVM::AllocaOp>(rewriter.getUnknownLoc(), LLVM::LLVMPointerType::get(funcOp->getContext()), graphStruct, rewriter.create<LLVM::ConstantOp>(rewriter.getUnknownLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(1)));

                                    if(arg->hasAttr("sym_name")){
                                        auto symNameAttr = arg->getAttrOfType<mlir::StringAttr>("sym_name");
                                        if(symNameAttr){
                                            alloc->setAttr("sym_name", rewriter.getStringAttr(symNameAttr.getValue()));
                                        }
                                    }

                                    if (arg->use_empty()) {
                                        toErase.push_back(arg);
                                    } else {
                                        arg->replaceAllUsesWith(alloc);
                                        toErase.push_back(arg);
                                    }

                                }

                                if(argType.isa<mlir::starplat::NodeType>()) {
                                    auto nodeStruct = createNodeStruct(&rewriter, funcOp->getContext());
                                    auto alloc = rewriter.create<LLVM::AllocaOp>(rewriter.getUnknownLoc(), LLVM::LLVMPointerType::get(funcOp->getContext()), nodeStruct, rewriter.create<LLVM::ConstantOp>(rewriter.getUnknownLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(1)));
                                    
                                    if (arg->hasAttr("sym_name")) {
                                        auto symNameAttr = arg->getAttrOfType<mlir::StringAttr>("sym_name");
                                        if (symNameAttr) {
                                            alloc->setAttr("sym_name", rewriter.getStringAttr(symNameAttr.getValue()));
                                        }
                                    }

                                    if (arg->use_empty()) {
                                        toErase.push_back(arg);
                                    } else {
                                        arg->replaceAllUsesWith(alloc);
                                        toErase.push_back(arg);
                                    }
                                    
                                }
                            }}); 
                        
                            for (mlir::Operation *op : toErase) {
                                op->erase();
                            }
                        
                        
                });
            }
        };
    }
}

mlir::LLVM::LLVMStructType createGraphStruct(mlir::IRRewriter *rewriter, mlir::MLIRContext *context)
{
    auto structType = LLVM::LLVMStructType::getIdentified(context, "Graph");
    structType.setBody({rewriter->getI32Type()}, false);

    return structType;
}

mlir::LLVM::LLVMStructType createNodeStruct(mlir::IRRewriter *rewriter, mlir::MLIRContext *context)
{
    auto structType = LLVM::LLVMStructType::getIdentified(context, "Node");
    structType.setBody({rewriter->getI32Type()}, false);

    return structType;
}
