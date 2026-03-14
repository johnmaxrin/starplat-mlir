#ifndef STARPLAT2OMP
#define STARPLAT2OMP

#include "includes/StarPlatOps.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/PassManager.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/LogicalResult.h"

#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Transforms/DialectConversion.h" // from @llvm-project

#include "typeconverter.h"

// mlir::Operation *generateGraphStruct(mlir::MLIRContext *context)
// {
//     auto structType = LLVM::LLVMStructType::getIdentified(context, "Graph");

//     return;
// }

inline mlir::LLVM::LLVMStructType createGraphStruct(mlir::MLIRContext* context) {

    auto structType = mlir::LLVM::LLVMStructType::getIdentified(context, "Graph");
    // Create Node struct type

    // auto ptrType = LLVM::LLVMPointerType::get(context);
    // Define Graph struct body with (Node*, int)
    // structType.setBody({ptrType, mlir::IntegerType::get(context, 32), ptrType}, /*isPacked=*/false);

    llvm::LogicalResult lr = structType.setBody({mlir::IntegerType::get(context, 64)}, /*isPacked=*/false);
    if (llvm::succeeded(lr))
        return structType;
    // structType.setBody({rewriter->getI32Type()}, false);

    return structType;
}

inline mlir::LLVM::LLVMStructType createNodeStruct(mlir::IRRewriter* rewriter, mlir::MLIRContext* context) {
    auto structType = mlir::LLVM::LLVMStructType::getIdentified(context, "Node");

    // Create a ptr type
    // auto ptr = mlir::LLVM::LLVMPointerType::get(rewriter->getI32Type());

    llvm::LogicalResult lr = structType.setBody({rewriter->getI32Type()}, false);
    if (llvm::succeeded(lr))
        return structType;

    return structType;
}

using namespace mlir;

struct ConvertFunc : public OpConversionPattern<mlir::starplat::FuncOp>
{
    using OpConversionPattern<mlir::starplat::FuncOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(mlir::starplat::FuncOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
        // llvm::errs() << "I came here!!\n";
        auto loc = op.getLoc();

        TypeConverter::SignatureConversion sigConversion(op.getArgNames()->size());

        for (auto [idx, argType] : llvm::enumerate(op.getArgumentTypes())) {
            auto convertedType = getTypeConverter()->convertType(argType);
            if (!convertedType) {
                llvm::errs() << "Invalid conversion\n";
                return failure();
            }
            sigConversion.addInputs(idx, convertedType);
        }

        SmallVector<Type> convertedResultTypes;
        if (failed(getTypeConverter()->convertTypes(op.getFunctionType().getResults(), convertedResultTypes)))
            return failure();
        // auto returnType = LLVM::LLVMPointerType::get(op.getContext());
        // // auto returnType =  MemRefType::get({mlir::ShapedType::kDynamic}, rewriter.getI1Type());
        //
        // // auto returnType = rewriter.getI64Type(); // up up
        //
        // // auto i64Type = rewriter.getI64Type();
        //
        // // Function signature: (i32, i32) -> i32
        auto oldFuncType = op.getFunctionType();
        //
        // // SmallVector<Type> paramTypes = {oldFuncType.getInput(0), oldFuncType.getInput(1), oldFuncType.getInput(2),
        // oldFuncType.getInput(3)};
        SmallVector<Type> paramTypes = {oldFuncType.getInput(0)};
        //
        // auto funcType = mlir::FunctionType::get(rewriter.getContext(), paramTypes, returnType);
        // auto funcType = FunctionType::get(rewriter.getContext(), {}, false);
        // auto llvmFuncTy = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(rewriter.getContext()), paramTypes, false);
        auto llvmFuncTy =
            LLVM::LLVMFunctionType::get(convertedResultTypes.empty() ? LLVM::LLVMVoidType::get(rewriter.getContext()) : convertedResultTypes.front(),
                                        sigConversion.getConvertedTypes(), false);

        auto funcName = op.getSymName();
        //
        // // Create the LLVM function
        auto funcOp2 = LLVM::LLVMFuncOp::create(rewriter, loc, funcName, llvmFuncTy);
        rewriter.inlineRegionBefore(op.getRegion(), funcOp2.getBody(), funcOp2.end());
        // // auto funcOp = LLVM::LLVMFuncOp::create(rewriter,loc, funcName, funcType);
        //
        // // Create an entry block with the right number of arguments
        // // Block *entryBlock = rewriter.createBlock(&funcOp.getBody(), {}, paramTypes, {loc, loc});
        //
        // Block& entryBlock = funcOp2.getBody().front();
        // for (BlockArgument arg : entryBlock.getArguments())
        //     assert(arg.use_empty() && "expected no uses of function arguments");
        //
        // llvm::BitVector argsToErase(entryBlock.getNumArguments(), true);
        // entryBlock.eraseArguments(argsToErase);
        //
        if (failed(rewriter.convertRegionTypes(&funcOp2.getBody(), *getTypeConverter(), &sigConversion)))
            return failure();

        // TODO: add attributes to generated function
        rewriter.eraseOp(op);
        // auto module = op->getParentOfType<mlir::ModuleOp>();
        // FailureOr<LLVM::LLVMFuncOp> graphAddEdgeFn = LLVM::lookupOrCreateFn(rewriter, module, "graph_add_edge", {}, rewriter.getI32Type());

        // Replace original starplat.func
        return success();
    }
};

struct ConvertDeclareOp : public OpConversionPattern<mlir::starplat::DeclareOp2>
{
    ConvertDeclareOp(mlir::MLIRContext* context) : OpConversionPattern<mlir::starplat::DeclareOp2>(context) {}

    using OpConversionPattern<mlir::starplat::DeclareOp2>::OpConversionPattern;

    LogicalResult matchAndRewrite(mlir::starplat::DeclareOp2 op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {

        auto resType = op->getResult(0).getType();
        if (isa<IntegerType>(resType)) {
            auto loc                   = op->getLoc();
            mlir::MLIRContext* context = getContext();
            auto allocaop              = LLVM::AllocaOp::create(rewriter, loc, LLVM::LLVMPointerType::get(context), rewriter.getI32Type(),
                                                                LLVM::ConstantOp::create(rewriter, loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(1)), 0);
            rewriter.replaceOp(op, allocaop);
        }
        else {
            llvm::outs() << "Error: This DeclareOp lowering not yet implemented.";
            exit(0);
        }
        // auto resType = op->getResult(0).getType();
        // if (isa<mlir::starplat::PropNodeType>(resType)) {
        //     auto loc          = op->getLoc();
        //
        //     auto rescast      = dyn_cast<mlir::starplat::PropNodeType>(resType);
        //
        //     auto field0       = LLVM::ExtractValueOp::create(rewriter, loc, mlir::IntegerType::get(op.getContext(), 64), adaptor.getOperands()[0],
        //                                                      rewriter.getDenseI64ArrayAttr({0}));
        //     Value dynamicSize = arith::IndexCastOp::create(rewriter, loc, rewriter.getIndexType(), field0);
        //
        //     MemRefType memrefType;
        //     if (rescast.getParameter() == mlir::IntegerType::get(rewriter.getContext(), 1))
        //         memrefType = MemRefType::get({mlir::ShapedType::kDynamic}, rewriter.getI1Type());
        //
        //     else if (rescast.getParameter() == mlir::IntegerType::get(rewriter.getContext(), 64))
        //         memrefType = MemRefType::get({mlir::ShapedType::kDynamic}, rewriter.getI64Type());
        //
        //     else {
        //         llvm::outs() << "Error: MemrefType not implemented\n";
        //         exit(0);
        //     }
        //     Value allocated = memref::AllocOp::create(rewriter, loc, memrefType, mlir::ValueRange({dynamicSize}));
        //
        //     rewriter.replaceOp(op, allocated);
        // }
        //
        // else if (isa<mlir::IntegerType>(resType)) {
        //     auto loc     = op->getLoc();
        //     auto resCast = dyn_cast<mlir::IntegerType>(resType);
        //
        //     MemRefType memrefType;
        //
        //     if (resCast.getWidth() == 64)
        //         memrefType = MemRefType::get({}, rewriter.getI64Type());
        //     else if (resCast.getWidth() == 1)
        //         memrefType = MemRefType::get({}, rewriter.getI1Type());
        //
        //     else {
        //         llvm::outs() << "Error: MemrefType Integer type not implemented.\n";
        //         exit(0);
        //     }
        //
        //     Value allocated = memref::AllocOp::create(rewriter, loc, memrefType);
        //     rewriter.replaceOp(op, allocated);
        // }
        //
        // else if (isa<mlir::starplat::NodeType>(resType)) {
        //     llvm::outs() << "Hello\n";
        // }
        //
        // else {
        //     llvm::outs() << "Error: This DeclareOp lowering not yet implemented.";
        //     return failure();
        // }

        return success();
    }
};

struct ConvertConstOp : public OpConversionPattern<mlir::starplat::ConstOp>
{
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(mlir::starplat::ConstOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {

        auto loc   = op.getLoc();

        auto value = op.getValueAttr();

        if (isa<mlir::IntegerAttr>(value)) {
            auto newOp = LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64Type(), value);
            rewriter.replaceOp(op, newOp);
        }
        else {
            llvm::outs() << "Error: This ConstantOp lowering not yet implemented.";
            exit(0);
        }

        // if (cast<mlir::StringAttr>(value).getValue() == "False") {
        // auto newOp = LLVM::ConstantOp::create(rewriter, loc, mlir::IntegerType::get(op.getContext(), 1), rewriter.getBoolAttr(0));
        //     rewriter.replaceOp(op, newOp);
        // }
        //
        // else if (cast<mlir::StringAttr>(value).getValue() == "True") {
        //     auto newOp = LLVM::ConstantOp::create(rewriter, loc, mlir::IntegerType::get(op.getContext(), 1), rewriter.getBoolAttr(1));
        //     rewriter.replaceOp(op, newOp);
        // }
        // else {
        //     llvm::outs() << "Error: This ConstantOp lowering not yet implemented.";
        //     exit(0);
        // }

        return success();
    }
};

namespace mlir
{
namespace starplat
{
#define GEN_PASS_DEF_CONVERTSTARPLATIRTOOMPPASS
#include "Passes.h.inc"

struct ConvertStarPlatIRToOMPPass : public mlir::starplat::impl::ConvertStarPlatIRToOMPPassBase<ConvertStarPlatIRToOMPPass>
{
    using ConvertStarPlatIRToOMPPassBase::ConvertStarPlatIRToOMPPassBase;

    void runOnOperation() override {

        mlir::MLIRContext* context = &getContext();
        auto* module               = getOperation();

        if (!module)
            llvm::errs() << "Module not found!\n";

        if (!context)
            llvm::errs() << "Context not found!\n";

        ConversionTarget target(getContext());

        target.addLegalDialect<mlir::LLVM::LLVMDialect>();
        target.addLegalDialect<mlir::scf::SCFDialect>();
        target.addLegalDialect<mlir::memref::MemRefDialect>();
        target.addLegalDialect<mlir::func::FuncDialect>();
        target.addLegalDialect<mlir::arith::ArithDialect>();

        // target.addIllegalOp<mlir::starplat::AddOp>();
        target.addIllegalOp<mlir::starplat::FuncOp>();
        target.addIllegalOp<mlir::starplat::DeclareOp>();
        // target.addIllegalOp<mlir::starplat::AttachNodePropertyOp>();
        // target.addIllegalOp<mlir::starplat::ConstOp>();
        // target.addIllegalOp<mlir::starplat::AssignmentOp>();
        // target.addIllegalOp<mlir::starplat::SetNodePropertyOp>();
        // target.addIllegalOp<mlir::starplat::FixedPointUntilOp>();

        RewritePatternSet patterns(context);
        StarPlatTypeConverter typeConverter(context);
        // StarplatToLLVMTypeConverter typeConverter(context);

        // patterns.add<ConvertAdd>(context);
        patterns.add<ConvertFunc>(typeConverter, context);
        patterns.add<ConvertDeclareOp>(typeConverter, context);
        patterns.add<ConvertConstOp>(typeConverter, context);
        // patterns.add<ConvertDeclareOp>(typeConverter, context);

        // populateFunctionOpInterfaceTypeConversionPattern<mlir::starplat::FuncOp>(patterns, typeConverter);
        // target.addDynamicallyLegalOp<mlir::starplat::FuncOp>([&](starplat::FuncOp op)
        //                                    {

        //     auto isSignatureLegal = typeConverter.isSignatureLegal(op.getFunctionType());
        //     auto isLegal = typeConverter.isLegal(&op.getBody());

        //     return isSignatureLegal && isLegal; });

        if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
            signalPassFailure();
        }
    }
};
} // namespace starplat
} // namespace mlir

#endif
