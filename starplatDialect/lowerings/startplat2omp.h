#include "mlir/Conversion/Passes.h"
#include "mlir/Pass/PassManager.h"

#include "includes/StarPlatDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/TypeID.h"
#include "llvm/Support/Casting.h"

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

        // addConversion([](Type type)
        //               { return type; });

        addConversion([ctx](mlir::starplat::GraphType graph) -> Type
                      {
                          MemRefType memrefType = MemRefType::get({4}, LLVM::LLVMPointerType::get(ctx));
                          return memrefType;
                          // return LLVM::LLVMPointerType::get(ctx);
                      });

        addTargetMaterialization(
            [ctx](OpBuilder &builder, Type type, ValueRange inputs, Location loc) -> std::optional<Value>
            {
                if (type.isa<mlir::starplat::GraphType>() &&
                    inputs.size() == 1 &&
                    inputs[0].getType().isa<MemRefType>())
                {
                    return inputs[0]; // just forward the memref
                }
                return std::nullopt;
            });

        addConversion([ctx](mlir::starplat::NodeType node) -> Type
                      { return LLVM::LLVMPointerType::get(ctx); });

        addConversion([ctx](mlir::IntegerType intType) -> Type
                      { return LLVM::LLVMPointerType::get(ctx); });

        addConversion([ctx](mlir::starplat::PropNodeType intType) -> Type
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
        auto oldFuncType = op.getFunctionType();

        // SmallVector<Type> paramTypes = {oldFuncType.getInput(0), oldFuncType.getInput(1), oldFuncType.getInput(2), oldFuncType.getInput(3)};
        SmallVector<Type> paramTypes = {oldFuncType.getInput(0)};

        auto funcType = mlir::FunctionType::get(rewriter.getContext(), paramTypes, returnType);
        auto funcName = op.getSymName();

        // Create the LLVM function
        auto funcOp2 = rewriter.create<mlir::func::FuncOp>(loc, funcName, funcType);
        // auto funcOp = rewriter.create<LLVM::LLVMFuncOp>(loc, funcName, funcType);

        // Create an entry block with the right number of arguments
        // Block *entryBlock = rewriter.createBlock(&funcOp.getBody(), {}, paramTypes, {loc, loc});

        rewriter.inlineRegionBefore(op.getRegion(), funcOp2.getBody(), funcOp2.end());
        rewriter.eraseOp(op);

        // Replace original starplat.func
        return success();
    }
};

struct ConvertAttachNode : public OpConversionPattern<mlir::starplat::AttachNodePropertyOp>
{
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(
        mlir::starplat::AttachNodePropertyOp op, OpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {

        auto arraySize = rewriter.create<LLVM::ConstantOp>(
            op.getLoc(),
            rewriter.getI32Type(),
            rewriter.getIntegerAttr(rewriter.getI32Type(), 1));

        auto func = op->getParentOp();

        rewriter.replaceOp(op.getOperation(), arraySize);
        // rewriter.eraseOp(op);

        return success();
    }
};

struct ConvertSetNodeProp : public OpConversionPattern<mlir::starplat::SetNodePropertyOp>
{
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(
        mlir::starplat::SetNodePropertyOp op, OpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {

        // auto arraySize = rewriter.create<LLVM::ConstantOp>(
        //     op.getLoc(),
        //     rewriter.getI32Type(),
        //     rewriter.getIntegerAttr(rewriter.getI32Type(), 1));

        // auto func = op->getParentOp();

        // rewriter.replaceOp(op.getOperation(), arraySize);
        rewriter.eraseOp(op);

        return success();
    }
};

struct ConvertAssignOp : public OpConversionPattern<mlir::starplat::AssignmentOp>
{
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(
        mlir::starplat::AssignmentOp op, OpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {

        // auto arraySize = rewriter.create<LLVM::ConstantOp>(
        //     op.getLoc(),
        //     rewriter.getI32Type(),
        //     rewriter.getIntegerAttr(rewriter.getI32Type(), 1));

        // auto func = op->getParentOp();

        // rewriter.replaceOp(op.getOperation(), arraySize);
        rewriter.eraseOp(op);

        return success();
    }
};

struct ConvertDeclareOp : public OpConversionPattern<mlir::starplat::DeclareOp>
{
    ConvertDeclareOp(mlir::MLIRContext *context)
        : OpConversionPattern<mlir::starplat::DeclareOp>(context) {}

    using OpConversionPattern<mlir::starplat::DeclareOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        mlir::starplat::DeclareOp op, OpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {

        auto elementType = rewriter.getI32Type(); // type of the element to allocate

        // optional array size (can be omitted or passed as a Value)

        auto resType = op->getResult(0).getType();
        if (resType.isa<mlir::starplat::PropNodeType>())
        {

            auto graph = op->getOperand(0);

            auto loc = op->getLoc();

            auto elementType = adaptor.getOperands()[0].getType().cast<mlir::MemRefType>().getElementType();
            // auto elementType = graph.getType().cast<MemRefType>().getElementType();
            auto zero = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
            //auto elementType = MemRefType::get({4}, LLVM::LLVMPointerType::get(op.getContext()));
            
  // static void build(::mlir::Type result, ::mlir::Value memref, ::mlir::ValueRange indices, /*optional*/::mlir::BoolAttr nontemporal);
            Value ptrToSize = rewriter.create<mlir::memref::LoadOp>(
                loc,
                elementType,
                adaptor.getOperands()[0],
                mlir::ValueRange(zero)
            );
            
            MemRefType memrefType = MemRefType::get({4}, rewriter.getF32Type());
            auto allocaOp = rewriter.create<memref::AllocaOp>(op.getLoc(), memrefType);

            rewriter.replaceOp(op, allocaOp.getOperation());

            op->getParentOp()->dump();
        }

        else
        {
            llvm::outs() << "Error: DeclareOp lowering not yet implemented.";
            return failure();
        }

        return success();
    }
};

struct ConvertConstOp : public OpConversionPattern<mlir::starplat::ConstOp>
{
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(
        mlir::starplat::ConstOp op, OpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {

        auto arraySize = rewriter.create<LLVM::ConstantOp>(
            op.getLoc(),
            rewriter.getI32Type(),
            rewriter.getIntegerAttr(rewriter.getI32Type(), 1));

        rewriter.replaceOp(op, arraySize);

        return success();
    }
};

struct ConvertFixedPointOp : public OpConversionPattern<mlir::starplat::FixedPointUntilOp>
{
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(
        mlir::starplat::FixedPointUntilOp op, OpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {

        // llvm::outs() << "FixedPoint Op Matched\n";

        //        auto func = op->getParentOp();
        //      func->dump();

        rewriter.eraseOp(op);

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
        // rewriter.eraseOp(op);

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
                target.addLegalDialect<mlir::scf::SCFDialect>();
                target.addLegalDialect<mlir::memref::MemRefDialect>();
                target.addLegalDialect<mlir::func::FuncDialect>();
                target.addLegalDialect<mlir::arith::ArithDialect>();

                // target.addIllegalOp<mlir::starplat::FuncOp>();
                target.addIllegalOp<mlir::starplat::AddOp>();
                target.addIllegalOp<mlir::starplat::DeclareOp>();
                target.addIllegalOp<mlir::starplat::AttachNodePropertyOp>();
                target.addIllegalOp<mlir::starplat::ConstOp>();
                target.addIllegalOp<mlir::starplat::AssignmentOp>();
                target.addIllegalOp<mlir::starplat::SetNodePropertyOp>();
                target.addIllegalOp<mlir::starplat::FixedPointUntilOp>();

                RewritePatternSet patterns(context);
                StarplatToLLVMTypeConverter typeConverter(context);

                patterns.add<ConvertAdd, ConvertDeclareOp, ConvertConstOp, ConvertSetNodeProp, ConvertAssignOp,
                             ConvertFunc, ConvertFixedPointOp, ConvertAttachNode>(context);

                populateFunctionOpInterfaceTypeConversionPattern<mlir::starplat::FuncOp>(patterns, typeConverter);
                target.addDynamicallyLegalOp<mlir::starplat::FuncOp>([&](starplat::FuncOp op)
                                                                     {

                    auto isSignatureLegal = typeConverter.isSignatureLegal(op.getFunctionType());
                    auto isLegal = typeConverter.isLegal(&op.getBody());

                    return isSignatureLegal && isLegal; });

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
