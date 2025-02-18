#include "includes/StarPlatDialect.h"
#include "includes/StarPlatOps.h"
#include "includes/StarPlatTypes.h"

#include "mlir/IR/Builders.h"  // For mlir::OpBuilder
#include "mlir/IR/Operation.h" // For mlir::Operation
#include "mlir/IR/Dialect.h"   // For mlir::Dialect

#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

class StarPlatCodeGen : public Visitor
{

public:
    StarPlatCodeGen() : result(0),
                        context(),
                        builder(&context),
                        module(mlir::ModuleOp::create(builder.getUnknownLoc()))
    {
        // Load Dialects here.
        context.getOrLoadDialect<mlir::starplat::StarPlatDialect>();
    }

    virtual void visitDeclarationStmt(const DeclarationStatement *dclstmt) override
    {
    }

    virtual void visitTemplateDeclarationStmt(const TemplateDeclarationStatement *templateDeclStmt)
    {
    }

    virtual void visitTemplateType(const TemplateType *templateType)
    {
    }

    virtual void visitForallStmt(const ForallStatement *forAllStmt) override
    {
    }

    virtual void visitIfStmt(const IfStatement *ifStmt) override
    {
    }

    virtual void visitBoolExpr(const BoolExpr *boolExpr) override
    {
    }

    virtual void visitIncandassignstmt(const Incandassignstmt *incandassignstmt) override
    {
    }

    virtual void visitIdentifier(const Identifier *identifier) override
    {
    }

    virtual void visitReturnStmt(const ReturnStmt *returnStmt) override
    {
    }

    virtual void visitParameterAssignment(const ParameterAssignment *paramAssignment)
    {
    }

    virtual void visitParam(const Param *param)
    {
    }

    virtual void visitTupleAssignment(const TupleAssignment *tupleAssignment)
    {
    }

    virtual void visitFunction(const Function *function) override
    {
        // Create function type. 
        llvm::SmallVector<mlir::Type, 4> argTypes;
        llvm::ArrayRef<llvm::StringRef> argNames;
        // TODO: Make Function return the number of arguments and change 4 to that number.


        auto args = function->getparams()->getArgList();
        for(auto arg : args)
        {
            if(arg->getType() != nullptr)
            {
                if(arg->getType()->getType() == "Graph")
                    argTypes.push_back(builder.getType<mlir::starplat::GraphType>()); 
            }
            else if(arg->getTemplateType() != nullptr)
            {
                if(arg->getTemplateType()->getGraphPropNode()->getPropertyType() == "propNode")
                    argTypes.push_back(builder.getType<mlir::starplat::PropNodeType>(builder.getI32Type()));
            }

            argNames.vec
            argNames.push_back(builder.getStringAttr(arg->getVarName()->getname()));
        }

        auto funcType = builder.getFunctionType(argTypes, {});
        auto argNames = builder.getStrArrayAttr(argNames(argNamesVec));

        
    }

    virtual void visitParamlist(const Paramlist *paramlist) override
    {
    }

    virtual void visitFixedpointUntil(const FixedpointUntil *fixedpointuntil) override
    {
    }

    virtual void visitInitialiseAssignmentStmt(const InitialiseAssignmentStmt *initialiseAssignmentStmt)
    {
    }

    virtual void visitMemberAccessAssignment(const MemberAccessAssignment *memberAccessAssignment)
    {
    }

    virtual void visitKeyword(const Keyword *keyword) override
    {
    }

    virtual void visitGraphProperties(const GraphProperties *graphproperties) override
    {
    }

    virtual void visitMethodcall(const Methodcall *methodcall) override
    {
    }

    virtual void visitMemberaccess(const Memberaccess *memberaccess) override
    {
    }

    virtual void visitArglist(const Arglist *arglist) override
    {
    }

    virtual void visitArg(const Arg *arg) override
    {
    }

    virtual void visitStatement(const Statement *statement) override
    {
    }

    virtual void visitStatementlist(const Statementlist *stmtlist) override
    {
    }

    virtual void visitType(const TypeExpr *type) override
    {
    }

    virtual void visitNumber(const Number *number) override
    {
    }

    virtual void visitExpression(const Expression *expr) override
    {
    }

    void print()
    {
        verify(module);
        module.dump();
    }

private:
    int result;
    map<string, int> variables;

    mlir::MLIRContext context;
    mlir::OpBuilder builder;
    mlir::ModuleOp module;
};