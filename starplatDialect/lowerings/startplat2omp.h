#include "mlir/Conversion/Passes.h"
#include "mlir/Pass/PassManager.h"

#include "includes/StarPlatDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "mlir/Transforms/DialectConversion.h" // from @llvm-project

// mlir::Operation *generateGraphStruct(mlir::MLIRContext *context)
// {
//     auto structType = LLVM::LLVMStructType::getIdentified(context, "Graph");

//     return;
// }

mlir::LLVM::LLVMStructType createGraphStruct(mlir::IRRewriter *rewriter, mlir::MLIRContext *context);
mlir::LLVM::LLVMStructType createNodeStruct(mlir::IRRewriter *rewriter, mlir::MLIRContext *context);

class StarplatToLLVMTypeConverter : public TypeConverter
{
public:
    StarplatToLLVMTypeConverter(MLIRContext *ctx)
    {
        addConversion([](Type type)
                      { return type; });

        addConversion([ctx](mlir::starplat::GraphType graph)
                      { return mlir::IntegerType::get(ctx, 32, mlir::IntegerType::SignednessSemantics::Signless); });

        addConversion([ctx](mlir::starplat::NodeType node)
                      { return mlir::IntegerType::get(ctx, 32, mlir::IntegerType::SignednessSemantics::Signless); });
    }
};

struct ConvertFunc : public OpConversionPattern<mlir::starplat::FuncOp>
{
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(
        mlir::starplat::FuncOp op, OpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        llvm::outs() << "Func Matched\n";
        auto loc = op.getLoc();
        auto returnType = rewriter.getI32Type();
        auto i32Type = rewriter.getI32Type();

        // Function signature: (i32, i32) -> i32
        auto oldFuncType = op.getFuncType();
        SmallVector<Type> paramTypes = {oldFuncType.getInput(0), oldFuncType.getInput(1), oldFuncType.getInput(2), oldFuncType.getInput(3)};
 
        
        auto funcType = LLVM::LLVMFunctionType::get(returnType, paramTypes, /*isVarArg=*/false);
        auto funcName = op.getSymName();

        // Create the LLVM function
        auto funcOp = rewriter.create<LLVM::LLVMFuncOp>(loc, funcName, funcType);

        // Create an entry block with the right number of arguments
        // Block *entryBlock = rewriter.createBlock(&funcOp.getBody(), {}, paramTypes, {loc, loc});

        rewriter.inlineRegionBefore(op.getRegion(), funcOp.getBody(), funcOp.end());
        rewriter.eraseOp(op);

        // Replace original starplat.func

        funcOp->dump();

        return success();
    }
};

struct ConvertAdd : public OpConversionPattern<mlir::starplat::AddOp>
{
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(
        mlir::starplat::AddOp op, OpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {

        llvm::outs() << "Add Matched\n";

        auto addOp = rewriter.create<LLVM::AddOp>(op.getLoc(), rewriter.getI32Type(), op->getOperand(0), op->getOperand(1));

        addOp.dump();

        rewriter.replaceOp(op.getOperation(), addOp);
        return success();
    }
};

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
                mlir::MLIRContext *context = &getContext();
                auto *module = getOperation();

                ConversionTarget target(*context);

                target.addLegalOp<mlir::LLVM::LLVMFuncOp>();
                target.addLegalOp<mlir::LLVM::AddOp>();

                target.addIllegalOp<mlir::starplat::FuncOp>();
                target.addIllegalOp<mlir::starplat::AddOp>();

                RewritePatternSet patterns(context);
                StarplatToLLVMTypeConverter typeConverter(context);
                patterns.add<ConvertAdd, ConvertFunc>(typeConverter, context);

                if (failed(applyPartialConversion(module, target, std::move(patterns))))
                {
                    signalPassFailure();
                }
            }
        };
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
