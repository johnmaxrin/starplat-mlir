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

    pm.addPass(mlir::starplat::createVertexToEdge());
    if (failed(pm.run(starplatcodegen->getModule()->getOperation()))) {
        llvm::errs() << "Failed to run passes\n";
        return 1;
    }

    

    // MLIRCodeGen *MLIRgen = new MLIRCodeGen;

    // if(root!= nullptr)
    //     root->Accept(MLIRgen);
    
    // MLIRgen->printModule();
    

    return 0;
}