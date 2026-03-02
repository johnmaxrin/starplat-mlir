#ifndef STARPLAT2OMP
#define STARPLAT2OMP

#include "includes/StarPlatOps.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Pass/PassManager.h"

#include "includes/StarPlatDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

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
#include "llvm/Support/LogicalResult.h"

#include "mlir/Transforms/DialectConversion.h" // from @llvm-project

// mlir::Operation *generateGraphStruct(mlir::MLIRContext *context)
// {
//     auto structType = LLVM::LLVMStructType::getIdentified(context, "Graph");

//     return;
// }

mlir::LLVM::LLVMStructType createGraphStruct(mlir::MLIRContext* context);
mlir::LLVM::LLVMStructType createNodeStruct(mlir::IRRewriter* rewriter, mlir::MLIRContext* context);
using namespace mlir;

struct ConvertFunc : public OpConversionPattern<mlir::starplat::FuncOp>
{
    using OpConversionPattern<mlir::starplat::FuncOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(mlir::starplat::FuncOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
        llvm::errs() << "I came here!!\n";
        auto loc        = op.getLoc();

        auto returnType = LLVM::LLVMPointerType::get(op.getContext());
        // auto returnType =  MemRefType::get({mlir::ShapedType::kDynamic}, rewriter.getI1Type());

        // auto returnType = rewriter.getI64Type(); // up up

        // auto i64Type = rewriter.getI64Type();

        // Function signature: (i32, i32) -> i32
        auto oldFuncType = op.getFunctionType();

        // SmallVector<Type> paramTypes = {oldFuncType.getInput(0), oldFuncType.getInput(1), oldFuncType.getInput(2), oldFuncType.getInput(3)};
        SmallVector<Type> paramTypes = {oldFuncType.getInput(0)};

        auto funcType                = mlir::FunctionType::get(rewriter.getContext(), paramTypes, returnType);
        auto funcName                = op.getSymName();

        // Create the LLVM function
        auto funcOp2 = mlir::func::FuncOp::create(rewriter, loc, funcName, funcType);
        // auto funcOp = LLVM::LLVMFuncOp::create(rewriter,loc, funcName, funcType);

        // Create an entry block with the right number of arguments
        // Block *entryBlock = rewriter.createBlock(&funcOp.getBody(), {}, paramTypes, {loc, loc});

        rewriter.inlineRegionBefore(op.getRegion(), funcOp2.getBody(), funcOp2.end());
        rewriter.eraseOp(op);

        // Replace original starplat.func
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

        target.addIllegalOp<mlir::starplat::AddOp>();
        // target.addIllegalOp<mlir::starplat::DeclareOp>();
        // target.addIllegalOp<mlir::starplat::AttachNodePropertyOp>();
        // target.addIllegalOp<mlir::starplat::ConstOp>();
        // target.addIllegalOp<mlir::starplat::AssignmentOp>();
        // target.addIllegalOp<mlir::starplat::SetNodePropertyOp>();
        // target.addIllegalOp<mlir::starplat::FixedPointUntilOp>();

        RewritePatternSet patterns(context);
        // StarplatToLLVMTypeConverter typeConverter(context);

        // patterns.add<ConvertAdd>(context);
        // patterns.add<ConvertFunc>(typeConverter, context);

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

mlir::LLVM::LLVMStructType createGraphStruct(mlir::MLIRContext* context) {

    auto structType = LLVM::LLVMStructType::getIdentified(context, "Graph");
    // Create Node struct type

    // auto ptrType = LLVM::LLVMPointerType::get(context);
    // Define Graph struct body with (Node*, int)
    // structType.setBody({ptrType, mlir::IntegerType::get(context, 32), ptrType}, /*isPacked=*/false);

    LogicalResult lr = structType.setBody({mlir::IntegerType::get(context, 64)}, /*isPacked=*/false);
    if (llvm::succeeded(lr))
        return structType;
    // structType.setBody({rewriter->getI32Type()}, false);

    return structType;
}

mlir::LLVM::LLVMStructType createNodeStruct(mlir::IRRewriter* rewriter, mlir::MLIRContext* context) {
    auto structType = LLVM::LLVMStructType::getIdentified(context, "Node");

    // Create a ptr type
    // auto ptr = mlir::LLVM::LLVMPointerType::get(rewriter->getI32Type());

    LogicalResult lr = structType.setBody({rewriter->getI32Type()}, false);
    if (llvm::succeeded(lr))
        return structType;

    return structType;
}

#endif
