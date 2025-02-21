// Next Iteration TODOs
// 1. Revmap the parser file. It's very messy.
// 2. Think if we need to cast here or at parser file.
// 3. Return String instead of const char *

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
#include <string>

class StarPlatCodeGen : public Visitor
{

public:
    StarPlatCodeGen() : result(0),
                        context(),
                        builder(&context),
                        module(mlir::ModuleOp::create(builder.getUnknownLoc())),
                        symbolTable(module)
    {
        // Load Dialects here.
        context.getOrLoadDialect<mlir::starplat::StarPlatDialect>();
    }

    virtual void visitDeclarationStmt(const DeclarationStatement *dclstmt) override
    {
    }

    virtual void visitTemplateDeclarationStmt(const TemplateDeclarationStatement *templateDeclStmt)
    {

        TemplateType *Type = static_cast<TemplateType *>(templateDeclStmt->gettype());
        Identifier *identifier = static_cast<Identifier *>(templateDeclStmt->getvarname());

        // TODO: Change the function name to getGraphProp().
        if (std::string(Type->getGraphPropNode()->getPropertyType()) == "propNode")
        {
            auto type = builder.getType<mlir::starplat::PropNodeType>(builder.getI32Type());
            auto typeAttr = ::mlir::TypeAttr::get(type);
            auto resType = builder.getI32Type();
            auto declare = builder.create<mlir::starplat::DeclareOp>(builder.getUnknownLoc(), resType, typeAttr, builder.getStringAttr(identifier->getname()));
            symbolTable.insert(declare);
        }

        else if (std::string(Type->getGraphPropNode()->getPropertyType()) == "propEdge")
        {
            auto type = builder.getType<mlir::starplat::PropEdgeType>(builder.getI32Type());
            auto typeAttr = ::mlir::TypeAttr::get(type);
            auto resType = builder.getI32Type();
            auto declare = builder.create<mlir::starplat::DeclareOp>(builder.getUnknownLoc(), resType, typeAttr, builder.getStringAttr(identifier->getname()));
            symbolTable.insert(declare);
        }
    }

    virtual void visitTemplateType(const TemplateType *templateType)
    {
    }

    virtual void visitForallStmt(const ForallStatement *forAllStmt) override
    {
    }

    virtual void visitMemberaccessStmt(const MemberacceessStmt *MemberacceessStmt) override
    {
        const Memberaccess *memberaccessnode = static_cast<const Memberaccess *>(MemberacceessStmt->getMemberAccess());
        const Methodcall *methodcallnode = static_cast<const Methodcall *>(memberaccessnode->getMethodCall());
        const Paramlist *paramlist = static_cast<const Paramlist *>(methodcallnode->getParamLists());

        if (methodcallnode->getIsBuiltin())
        {
            if (std::strcmp(methodcallnode->getIdentifier()->getname(), "attachNodeProperty") == 0)
            {
                // Create attachNodeProperty operation.
                llvm::outs() << methodcallnode->getIdentifier()->getname() << "\n";
                // auto attachNodeProperty = builder.create<mlir::starplat::AttachNodePropertyOp>(builder.getUnknownLoc(), builder.getStringAttr("Hello World"));
            }

            else
            {
                llvm::errs() << methodcallnode->getIdentifier()->getname() << " is not implemented yet\n";
            }
        }

        methodcallnode->Accept(this);
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
        Identifier *identifier = static_cast<Identifier *>(paramAssignment->getidentifier());
        Keyword *keyword = static_cast<Keyword *>(paramAssignment->getkeyword());

        if (!symbolTable.lookup(keyword->getKeyword()))
        {
            auto keywordVal = builder.create<mlir::starplat::ConstOp>(builder.getUnknownLoc(), builder.getI32Type(), builder.getStringAttr(std::string(keyword->getKeyword())), builder.getStringAttr(keyword->getKeyword()));

            if (keywordVal)
                symbolTable.insert(keywordVal);
            else
            {
                llvm::outs() << "error: " << "while adding to Symbol Table\n";
                exit(1);
            }
        }
        if (symbolTable.lookup(identifier->getname()))
        {
            auto lhs = symbolTable.lookup(identifier->getname());
            auto rhs = symbolTable.lookup(keyword->getKeyword());

            auto assign = builder.create<mlir::starplat::AssignmentOp>(builder.getUnknownLoc(), lhs->getResult(0), rhs->getResult(0));
            //symbolTable.rename(assign, builder.getStringAttr(identifier->getname()));

        }
        // else
        // {
        //     llvm::outs() <<"error: " << identifier->getname() << " not declared\n";
        // }
    }

    virtual void visitParam(const Param *param)
    {
        const ParameterAssignment *paramAssignment = static_cast<const ParameterAssignment *>(param->getParamAssignment());
        if (paramAssignment != nullptr)
        {
            paramAssignment->Accept(this);
        }
    }

    virtual void visitTupleAssignment(const TupleAssignment *tupleAssignment)
    {
    }

    virtual void visitFunction(const Function *function) override
    {
        // Create function type.
        llvm::SmallVector<mlir::Type, 4> argTypes;
        llvm::SmallVector<mlir::Attribute> argNames;
        // TODO: Make Function return the number of arguments and change 4 to that number.

        auto args = function->getparams()->getArgList();
        for (auto arg : args)
        {
            if (arg->getType() != nullptr)
            {
                if (std::string(arg->getType()->getType()) == "Graph")
                {
                    argTypes.push_back(builder.getType<mlir::starplat::GraphType>());
                    auto GraphType = mlir::starplat::GraphType::get(builder.getContext());
                    auto declareArg = builder.create<mlir::starplat::DeclareOp>(builder.getUnknownLoc(), builder.getI32Type(), ::mlir::TypeAttr::get(GraphType), builder.getStringAttr(arg->getVarName()->getname()));
                    symbolTable.insert(declareArg);
                }
            }
            else if (arg->getTemplateType() != nullptr)
            {
                if (std::string(arg->getTemplateType()->getGraphPropNode()->getPropertyType()) == "propNode")
                {
                    argTypes.push_back(builder.getType<mlir::starplat::PropNodeType>(builder.getI32Type()));
                    auto type = builder.getType<mlir::starplat::PropNodeType>(builder.getI32Type());
                    auto typeAttr = ::mlir::TypeAttr::get(type);
                    auto declareArg = builder.create<mlir::starplat::DeclareOp>(builder.getUnknownLoc(), builder.getI32Type(), typeAttr, builder.getStringAttr(arg->getVarName()->getname()));
                    symbolTable.insert(declareArg);
                }
            }

            argNames.push_back(builder.getStringAttr(arg->getVarName()->getname()));
        }

        auto funcType = builder.getFunctionType(argTypes, {});
        mlir::ArrayAttr argNamesAttr = builder.getArrayAttr(argNames);

        auto func = builder.create<mlir::starplat::FuncOp>(builder.getUnknownLoc(), function->getfuncNameIdentifier(), funcType, argNamesAttr);

        module.push_back(func);
        auto &entryBlock = func.getBody().emplaceBlock();

        for (auto arg : funcType.getInputs())
            auto argval = entryBlock.addArgument(arg, builder.getUnknownLoc());

        builder.setInsertionPointToStart(&entryBlock);

        // Visit the function body.
        Statementlist *stmtlist = static_cast<Statementlist *>(function->getstmtlist());
        stmtlist->Accept(this);

        // Create end operation.
        auto end = builder.create<mlir::starplat::endOp>(builder.getUnknownLoc());
    }

    virtual void visitParamlist(const Paramlist *paramlist) override
    {
        vector<Param *> paramListVecvtor = paramlist->getParamList();

        for (Param *param : paramListVecvtor)
        {
            param->Accept(this);
        }
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
        const Paramlist *paramlist = static_cast<const Paramlist *>(methodcall->getParamLists());
        paramlist->Accept(this);
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
        for (ASTNode *stmt : stmtlist->getStatementList())
        {
            stmt->Accept(this);
        }
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
    mlir::SymbolTable symbolTable;
};