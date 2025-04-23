#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "ast/ast.h"
#include "ast/visitor.h"
#include "codegen/astdump.h"
#include "codegen/starplatIR.h"
#include "avial.tab.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include "tblgen2/StarPlatOps.cpp.inc"

#include "mlir/Pass/PassManager.h"
#include "transforms/reachingDef.h"
#include "transforms/vertexToEdge.h"

#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"


#include "lowerings/startplat2omp.h"

#define DEBUG_TYPE "dialect-conversion"

extern int yyparse();
extern FILE *yyin;
ASTNode* root;

int main(int argc, char *argv[])
{
    root = nullptr;

    if (argc < 2)
    {
        printf("%s usage\n%s <file name>\n", argv[0], argv[0]);
        return 0;
    }

    FILE *file = fopen(argv[1], "r");
    if (!file)
    {
        printf("Cannot open file. \n");
        return 0;
    }

    yyin = file;
    yyparse();
    fclose(file);

    
    // CodeGen *codegen = new CodeGen;

    // if(root != nullptr)
    //     root->Accept(codegen);

    StarPlatCodeGen *starplatcodegen = new StarPlatCodeGen;
    if(root!= nullptr)
        root->Accept(starplatcodegen, starplatcodegen->getSymbolTable());

    starplatcodegen->print();

    PassManager pm(starplatcodegen->getContext());

    //pm.addPass(mlir::starplat::createVertexToEdge());
    //pm.addPass(mlir::starplat::createReachDef());

    pm.addPass(mlir::starplat::createConvertStartPlatIRToOMPPass());

    if (failed(pm.run(starplatcodegen->getModule()->getOperation()))) {
        llvm::errs() << "Failed to run passes\n";
        return 1;
    }

    llvm::outs() << "\n\n\n\nAfter the pass:\n";
    starplatcodegen->print();


    llvm::outs() << "\n\n\n\nLower to LLVM IR:\n";
    PassManager pmllvm(starplatcodegen->getContext());
    pmllvm.addPass(mlir::createConvertFuncToLLVMPass());
    pmllvm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
    pmllvm.addPass(mlir::createReconcileUnrealizedCastsPass());

    if (failed(pmllvm.run(starplatcodegen->getModule()->getOperation()))) {
        llvm::errs() << "Failed to run passes\n";
        return 1;
    }
    
    starplatcodegen->print();

    // Work on Conversion of OMP
    // Working on Generating a hello world program in LLVM - Done 
    // Workign on generating OMP - Done 
    // Donw with the Blog and Repo

    // MLIRCodeGen *MLIRgen = new MLIRCodeGen;

    // if(root!= nullptr)
    //     root->Accept(MLIRgen);
    
    // MLIRgen->printModule();
    

    return 0;
}