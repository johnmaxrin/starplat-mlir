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

        addConversion([ctx](mlir::starplat::GraphType graph) -> Type
                      { return mlir::IntegerType::get(ctx, 32, mlir::IntegerType::SignednessSemantics::Signless); });

        addConversion([ctx](mlir::starplat::NodeType node)
                      { return mlir::IntegerType::get(ctx, 32, mlir::IntegerType::SignednessSemantics::Signless); });

        addConversion([ctx](mlir::IntegerType intType) -> Type
                      { return LLVM::LLVMPointerType::get(ctx); });
    }
};

struct ConvertFunc : public OpConversionPattern<mlir::starplat::FuncOp>
{
    using OpConversionPattern<mlir::starplat::FuncOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        mlir::starplat::FuncOp op, OpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
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
        return success();
    }
};

struct ConvertDeclareOp : public OpConversionPattern<mlir::starplat::DeclareOp>
{
    using OpConversionPattern<mlir::starplat::DeclareOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        mlir::starplat::DeclareOp op, OpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {

        llvm::outs() << "Declare Op Matched\n";

        auto elementType = rewriter.getI32Type(); // type of the element to allocate
        auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());

        // optional array size (can be omitted or passed as a Value)
        auto arraySize = rewriter.create<LLVM::ConstantOp>(
            op.getLoc(),
            rewriter.getI32Type(),
            rewriter.getIntegerAttr(rewriter.getI32Type(), 1));

        // Now create the AllocaOp
        // auto alloca = rewriter.create<LLVM::AllocaOp>(
        //     op.getLoc(),
        //     rewriter.getI32Type(),
        //     elementType,
        //     arraySize);

        // alloca.dump();
        
        rewriter.replaceOp(op, arraySize.getOperation());
        

        auto func = op->getParentOp();
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

        auto addOp = rewriter.create<LLVM::AddOp>(op.getLoc(), rewriter.getI32Type(), op->getOperand(0), op->getOperand(1));

        auto func = op->getParentOp();

        rewriter.replaceOp(op.getOperation(), addOp);
        //rewriter.eraseOp(op);

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

                ConversionTarget target(getContext());

                target.addLegalDialect<mlir::LLVM::LLVMDialect>();

                target.addIllegalOp<mlir::starplat::FuncOp>();
                target.addIllegalOp<mlir::starplat::AddOp>();
                // target.addIllegalOp<mlir::starplat::DeclareOp>();

                RewritePatternSet patterns(context);
                StarplatToLLVMTypeConverter typeConverter(context);
                patterns.add<ConvertAdd, ConvertFunc, ConvertDeclareOp>(context);

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
