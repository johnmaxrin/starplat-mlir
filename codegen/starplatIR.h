// Next Iteration TODOs
// 1. Revmap the parser file. It's very messy.
// 2. Think if we need to cast here or at parser file.
// 3. Return String instead of const char *
// 4. Remove ArgList Code gen from Funntion and write it in VisitArgList.
// 5. Check if the type is node propert inorder to do attachnode.
// 6. Add return to the Accept Function.
// 7. Rewrite visitForAllStmt.
// 8. Add switch instead of IF-ELSE.
// 9. Change edge -> Edge type.
// 10. Handle all the special function seperately.
// 11. Rewrite Add to Arithematic OP

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

class StarPlatCodeGen : public MLIRVisitor
{

public:
    StarPlatCodeGen() : context(),
                        builder(&context),
                        module(mlir::ModuleOp::create(builder.getUnknownLoc())),
                        globalSymbolTable(module)
    {
        // Load Dialects here.
        context.getOrLoadDialect<mlir::starplat::StarPlatDialect>();
    }

    virtual void visitDeclarationStmt(const DeclarationStatement *dclstmt, mlir::SymbolTable *symbolTable) override
    {
    }

    virtual void visitTemplateDeclarationStmt(const TemplateDeclarationStatement *templateDeclStmt, mlir::SymbolTable *symbolTable) override
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
            symbolTable->insert(declare);
        }

        else if (std::string(Type->getGraphPropNode()->getPropertyType()) == "propEdge")
        {
            auto type = builder.getType<mlir::starplat::PropEdgeType>(builder.getI32Type());
            auto typeAttr = ::mlir::TypeAttr::get(type);
            auto resType = builder.getI32Type();
            auto declare = builder.create<mlir::starplat::DeclareOp>(builder.getUnknownLoc(), resType, typeAttr, builder.getStringAttr(identifier->getname()));
            symbolTable->insert(declare);
        }
    }

    virtual void visitTemplateType(const TemplateType *templateType, mlir::SymbolTable *symbolTable) override
    {
    }

    virtual void visitForallStmt(const ForallStatement *forAllStmt, mlir::SymbolTable *symbolTable) override
    {
        Identifier *loopVar = static_cast<Identifier *>(forAllStmt->getLoopVar());
        const Expression *expr = static_cast<const Expression *>(forAllStmt->getexpr());
        const Statementlist *stmtlist = static_cast<const Statementlist *>(forAllStmt->getstmtlist());

        auto loopVarSymbol = symbolTable->lookup(loopVar->getname());
        mlir::Type loopVarType;
        mlir::Operation *loopVarOp;
        mlir::SmallVector<mlir::Attribute> loopAttr;
        mlir::SmallVector<mlir::Value> loopOperands;

        if (loopVarSymbol)
        {
            llvm::outs() << "Error: Identifier '" << loopVar->getname() << "' already in declared.\n";
            return;
        }

        // get the type of expr inorder to get the type of loopVar.
        if (expr->getKind() == ExpressionKind::KIND_MEMBERACCESS)
        {
            const Memberaccess *memberaccess = static_cast<const Memberaccess *>(expr->getExpression());
            const Methodcall *outermethodcall = static_cast<const Methodcall *>(memberaccess->getMethodCall()); // Filter Methodcall.

            const Paramlist *paramlisttz = static_cast<const Paramlist *>(outermethodcall->getParamLists());

            // Check if this is with nested access.
            // TODO: Do this recursively!
            if (memberaccess->getMemberAccessNode())
            {

                const Memberaccess *nestedMemberaccess = static_cast<const Memberaccess *>(memberaccess->getMemberAccessNode());
                const Identifier *identifier = nestedMemberaccess->getIdentifier(); // g
                const Methodcall *innerMethodcall = static_cast<const Methodcall *>(nestedMemberaccess->getMethodCall());
                auto idSymbol = symbolTable->lookup(identifier->getname());

                if (!idSymbol)
                {
                    llvm::outs() << "Error: Identifier '" << identifier->getname() << "' not declared.\n";
                    return;
                }

                loopOperands.push_back(idSymbol->getResult(0));

                const Identifier *innerMethodcallIdentifier = innerMethodcall->getIdentifier();

                if (strcmp(innerMethodcallIdentifier->getname(), "nodes") == 0)
                {
                    loopVarType = mlir::starplat::NodeType::get(builder.getContext());
                    loopVarOp = builder.create<mlir::starplat::DeclareOp>(builder.getUnknownLoc(), builder.getI32Type(), mlir::TypeAttr::get(loopVarType), builder.getStringAttr(loopVar->getname()));
                    symbolTable->insert(loopVarOp);
                    loopOperands.push_back(loopVarOp->getResult(0));

                    loopAttr.push_back(builder.getStringAttr("nodes"));
                }

                else
                {
                    llvm::outs() << "Error: Methodcall '" << innerMethodcallIdentifier->getname() << "' not Implemented.\n";
                    return;
                }
            }

            else if (memberaccess->getMethodCall())
            {
                const Methodcall *methodcallin = static_cast<const Methodcall *>(memberaccess->getMethodCall());
                if (methodcallin->getIsBuiltin())
                {
                    if (strcmp(methodcallin->getIdentifier()->getname(), "neighbors") == 0)
                    {
                        loopVarType = mlir::starplat::NodeType::get(builder.getContext());
                        loopVarOp = builder.create<mlir::starplat::DeclareOp>(builder.getUnknownLoc(), builder.getI32Type(), mlir::TypeAttr::get(loopVarType), builder.getStringAttr(loopVar->getname()));
                        symbolTable->insert(loopVarOp);
                        loopOperands.push_back(loopVarOp->getResult(0));
                        loopAttr.push_back(builder.getStringAttr("neighbours"));
                    }
                }
                else
                {
                    llvm::outs() << "Undefined method call: " << methodcallin->getIdentifier()->getname();
                    return;
                }
            }

            if (outermethodcall->getIsBuiltin())
            {

                const Identifier *identifier1 = static_cast<const Identifier *>(outermethodcall->getIdentifier());
                if (strcmp(identifier1->getname(), "filter") == 0)
                {

                    loopAttr.push_back(builder.getStringAttr("filter"));

                    const Expression *outer = static_cast<const Expression *>(outermethodcall->getParamLists());
                    const BoolExpr *outerBoolExpr = static_cast<const BoolExpr *>(outer->getExpression());

                    const Expression *lhsExpr = static_cast<const Expression *>(outerBoolExpr->getExpr1());
                    const Expression *rhsExpr = static_cast<const Expression *>(outerBoolExpr->getExpr2());
                    const char *op = outerBoolExpr->getop();

                    if (strcmp(op, "==") == 0)
                        loopAttr.push_back(builder.getStringAttr("EQS"));
                    else
                    {
                        llvm::outs() << "Error: Operator not implemented.\n";
                        return;
                    }

                    if (lhsExpr->getKind() == ExpressionKind::KIND_IDENTIFIER && rhsExpr->getKind() == ExpressionKind::KIND_KEYWORD)
                    {
                        const Identifier *lhsIdentifier = static_cast<const Identifier *>(lhsExpr->getExpression());
                        const Keyword *rhsKeyword = static_cast<const Keyword *>(rhsExpr->getExpression());

                        auto lhsidSymbol = symbolTable->lookup(lhsIdentifier->getname());
                        auto rhsKeywordSymbol = symbolTable->lookup(rhsKeyword->getKeyword());

                        if (!lhsidSymbol || !rhsKeywordSymbol)
                        {
                            llvm::outs() << "Error: Identifier '" << lhsIdentifier->getname() << "' or Keyword '" << rhsKeyword->getKeyword() << "' not declared.\n";
                            return;
                        }

                        loopOperands.push_back(lhsidSymbol->getResult(0));
                        loopOperands.push_back(rhsKeywordSymbol->getResult(0));
                    }

                    else
                    {
                        llvm::outs() << "Error: Not implemented. Syntax Error\n";
                        return;
                    }
                }
            }
        }

        mlir::ArrayAttr loopAttrArray = builder.getArrayAttr(loopAttr);
        auto loopOp = builder.create<mlir::starplat::ForAllOp>(builder.getUnknownLoc(), loopOperands, loopAttrArray);

        auto &loopBlock = loopOp.getBody().emplaceBlock();
        builder.setInsertionPointToStart(&loopBlock);
        stmtlist->Accept(this, symbolTable);

        builder.create<mlir::starplat::endOp>(builder.getUnknownLoc());
        builder.setInsertionPointAfter(loopOp);
    }

    virtual void visitMemberaccessStmt(const MemberacceessStmt *MemberacceessStmt, mlir::SymbolTable *symbolTable) override
    {
        const Memberaccess *memberaccessnode = static_cast<const Memberaccess *>(MemberacceessStmt->getMemberAccess());
        const Methodcall *methodcallnode = static_cast<const Methodcall *>(memberaccessnode->getMethodCall());
        const Paramlist *paramlist = static_cast<const Paramlist *>(methodcallnode->getParamLists());

        methodcallnode->Accept(this, symbolTable);
        vector<Param *> paramListVecvtor = paramlist->getParamList();

        const Identifier *identifier1 = memberaccessnode->getIdentifier();
        const Identifier *identifier2 = memberaccessnode->getIdentifier2();

        if (methodcallnode && methodcallnode->getIsBuiltin())
        {
            if (std::strcmp(methodcallnode->getIdentifier()->getname(), "attachNodeProperty") == 0)
            {
                // Create attachNodeProperty operation.

                llvm::SmallVector<mlir::Value> operandsForAttachNodeProperty;

                for (Param *param : paramListVecvtor)
                {
                    if (!param->getParamAssignment())
                        continue;

                    const auto *paramAssignment = static_cast<const ParameterAssignment *>(param->getParamAssignment());
                    if (!paramAssignment)
                        continue;

                    const auto *identifier = static_cast<const Identifier *>(paramAssignment->getidentifier());
                    const auto *keyword = static_cast<const Keyword *>(paramAssignment->getkeyword());
                    keyword->Accept(this, symbolTable);

                    if (!identifier || !keyword)
                    {
                        llvm::outs() << "error: " << "identifier or keyword is null\n";
                        exit(1);
                    }

                    auto *idSymbol = symbolTable->lookup(identifier->getname());
                    auto *kwSymbol = symbolTable->lookup(keyword->getKeyword());

                    if (!kwSymbol)
                        llvm::outs() << "Error: Keyword '" << keyword->getKeyword() << "' not declared.\n";

                    if (!idSymbol)
                        llvm::outs() << "Error: Identifier '" << identifier->getname() << "' not declared.\n";

                    if (idSymbol && kwSymbol)
                        operandsForAttachNodeProperty.push_back(idSymbol->getResult(0));
                    else
                        return; // Handle errors gracefully
                }

                auto attachNodeProp = builder.create<mlir::starplat::AttachNodePropertyOp>(builder.getUnknownLoc(), operandsForAttachNodeProperty);
            }

            else
            {
                llvm::errs() << methodcallnode->getIdentifier()->getname() << " is not implemented yet\n";
            }
        }
    }

    virtual void visitIfStmt(const IfStatement *ifStmt, mlir::SymbolTable *symbolTable) override
    {
    }

    virtual void visitBoolExpr(const BoolExpr *boolExpr, mlir::SymbolTable *symbolTable) override
    {
    }

    virtual void visitIncandassignstmt(const Incandassignstmt *incandassignstmt, mlir::SymbolTable *symbolTable) override
    {
    }

    virtual void visitIdentifier(const Identifier *identifier, mlir::SymbolTable *symbolTable) override
    {
    }

    virtual void visitReturnStmt(const ReturnStmt *returnStmt, mlir::SymbolTable *symbolTable) override
    {
    }

    virtual void visitParameterAssignment(const ParameterAssignment *paramAssignment, mlir::SymbolTable *symbolTable)
    {
        Identifier *identifier = static_cast<Identifier *>(paramAssignment->getidentifier());
        Keyword *keyword = static_cast<Keyword *>(paramAssignment->getkeyword());

        if (!symbolTable->lookup(keyword->getKeyword()))
        {
            auto keywordVal = builder.create<mlir::starplat::ConstOp>(builder.getUnknownLoc(), builder.getI32Type(), builder.getStringAttr(std::string(keyword->getKeyword())), builder.getStringAttr(keyword->getKeyword()));

            if (keywordVal)
                symbolTable->insert(keywordVal);
            else
            {
                llvm::outs() << "error: " << "while adding to Symbol Table\n";
                exit(1);
            }
        }
        if (symbolTable->lookup(identifier->getname()))
        {
            auto lhs = symbolTable->lookup(identifier->getname());
            auto rhs = symbolTable->lookup(keyword->getKeyword());

            auto assign = builder.create<mlir::starplat::AssignmentOp>(builder.getUnknownLoc(), lhs->getResult(0), rhs->getResult(0));
            // symbolTable->rename(assign, builder.getStringAttr(identifier->getname()));
        }
        else
        {
            llvm::outs() << "error: " << identifier->getname() << " not declared\n";
            exit(1);
        }
    }

    virtual void visitParam(const Param *param, mlir::SymbolTable *symbolTable)
    {
        const ParameterAssignment *paramAssignment = static_cast<const ParameterAssignment *>(param->getParamAssignment());
        if (paramAssignment != nullptr)
        {
            paramAssignment->Accept(this, symbolTable);
        }
    }

    virtual void visitTupleAssignment(const TupleAssignment *tupleAssignment, mlir::SymbolTable *symbolTable)
    {
        // <nbr.dist,nbr.modified_nxt> =
        //      <Min (nbr.dist, v.dist + e.weight), True>;

        // 4 Expressions
        // 1 - Member Access
        // 2 - Member Access
        // 3 - Method Call
        // 4 - Keyword

        const Expression *lhsexpr1 = static_cast<const Expression *>(tupleAssignment->getlhsexpr1());
        const Expression *lhsexpr2 = static_cast<const Expression *>(tupleAssignment->getlhsexpr2());
        const Expression *rhsexpr1 = static_cast<const Expression *>(tupleAssignment->getrhsexpr1());
        const Expression *rhsexpr2 = static_cast<const Expression *>(tupleAssignment->getrhsexpr2());

        mlir::Operation *operand1;
        mlir::Operation *operand2;
        mlir::Operation *operand3;
        mlir::Operation *operand4;

        if (lhsexpr1->getKind() == ExpressionKind::KIND_MEMBERACCESS)
        {
            const Memberaccess *lhs1MemberAccess = static_cast<const Memberaccess *>(lhsexpr1->getExpression());
            if (lhs1MemberAccess->getIdentifier() && lhs1MemberAccess->getIdentifier2())
            {
                if (symbolTable->lookup(lhs1MemberAccess->getIdentifier2()->getname()))
                {
                    if (symbolTable->lookup(lhs1MemberAccess->getIdentifier()->getname()))
                    {
                        mlir::Operation *propOp = symbolTable->lookup(lhs1MemberAccess->getIdentifier2()->getname());
                        mlir::Operation *varOp = symbolTable->lookup(lhs1MemberAccess->getIdentifier()->getname());
                        operand1 = builder.create<mlir::starplat::GetNodePropertyOp>(builder.getUnknownLoc(), builder.getI32Type(), varOp->getResult(0), propOp->getAttrOfType<mlir::StringAttr>("sym_name"));
                    }
                    else
                    {
                        llvm::outs() << "Error: Undefined Var `" << lhs1MemberAccess->getIdentifier()->getname() << "'/n.";
                        return;
                    }
                }
                else
                {
                    llvm::outs() << "Error: Undefined Property `" << lhs1MemberAccess->getIdentifier2()->getname() << "'/n.";
                    return;
                }
            }
            else
            {
                llvm::outs() << "Error: Tuple Assignment failed.\n";
                return;
            }
        }

        else
        {
            llvm::outs() << "Error: Tuple Assignment failed.\n";
            return;
        }

        if (lhsexpr2->getKind() == ExpressionKind::KIND_MEMBERACCESS)
        {
            const Memberaccess *lhs2MemberAccess = static_cast<const Memberaccess *>(lhsexpr2->getExpression());
            if (lhs2MemberAccess->getIdentifier() && lhs2MemberAccess->getIdentifier2())
            {
                if (symbolTable->lookup(lhs2MemberAccess->getIdentifier2()->getname()))
                {
                    if (symbolTable->lookup(lhs2MemberAccess->getIdentifier()->getname()))
                    {
                        mlir::Operation *propOp = symbolTable->lookup(lhs2MemberAccess->getIdentifier2()->getname());
                        mlir::Operation *varOp = symbolTable->lookup(lhs2MemberAccess->getIdentifier()->getname());
                        operand1 = builder.create<mlir::starplat::GetNodePropertyOp>(builder.getUnknownLoc(), builder.getI32Type(), varOp->getResult(0), propOp->getAttrOfType<mlir::StringAttr>("sym_name"));
                    }
                    else
                    {
                        llvm::outs() << "Error: Undefined Var `" << lhs2MemberAccess->getIdentifier()->getname() << "'/n.";
                        return;
                    }
                }
                else
                {
                    llvm::outs() << "Error: Undefined Property `" << lhs2MemberAccess->getIdentifier2()->getname() << "'/n.";
                    return;
                }
            }
            else
            {
                llvm::outs() << "Error: Tuple Assignment failed.\n";
                return;
            }
        }
        else
        {
            llvm::outs() << "Error: Tuple Assignment failed.\n";
            return;
        }

        if (rhsexpr1->getKind() == ExpressionKind::KIND_METHODCALL)
        {
            const Methodcall *rhsMethodCall = static_cast<const Methodcall *>(rhsexpr1->getExpression());
            if (rhsMethodCall->getIsBuiltin() && strcmp(rhsMethodCall->getIdentifier()->getname(), "Min") == 0)
            {
                // Visit Min Method.
                const Paramlist *rhsParamList = static_cast<const Paramlist *>(rhsMethodCall->getParamLists());
                rhsParamList->Accept(this, symbolTable);

                const Expression *operand1 = static_cast<const Expression *>(rhsParamList->getParamList()[0]->getExpr());
                const Expression *operand2 = static_cast<const Expression *>(rhsParamList->getParamList()[1]->getExpr());

                // Handle Operand1
                if (operand1->getKind() == ExpressionKind::KIND_MEMBERACCESS)
                {
                    const Memberaccess *operand1MemAccess = static_cast<const Memberaccess *>(operand1->getExpression());
                    const Identifier *id1 = static_cast<const Identifier *>(operand1MemAccess->getIdentifier());
                    const Identifier *id2 = static_cast<const Identifier *>(operand1MemAccess->getIdentifier2());

                    if (id1 && id2)
                    {
                        if (!symbolTable->lookup(id1->getname()))
                        {
                            llvm::outs() << id1->getname() << " not defined.\n";
                            exit(0);
                        }
                        if (!symbolTable->lookup(id2->getname()))
                        {
                            llvm::outs() << id2->getname() << " not defined.\n";
                            exit(0);
                        }

                        auto id1Op = symbolTable->lookup(id1->getname());
                        auto id2Op = symbolTable->lookup(id2->getname());

                        auto typeAttr = id1Op->getAttrOfType<mlir::TypeAttr>("type");
                        mlir::Type id1Optype = typeAttr.getValue();

                        if (id1Optype.isa<mlir::starplat::NodeType>())
                        {
                            // Generate get node property.
                            llvm::StringRef nameRef(id2->getname());
                            auto getnbrdist = builder.create<mlir::starplat::GetNodePropertyOp>(builder.getUnknownLoc(), builder.getI32Type(), id1Op->getResult(0), builder.getStringAttr(nameRef));
                        }
                    }
                    else
                    {
                        llvm::outs() << "Error: Tuple Assignment Error.\n";
                        return;
                    }
                }
                else
                {
                    llvm::outs() << "Error: Not implemented @ Tuple Assignment.\n";
                    return;
                }

                if (operand2->getKind() == ExpressionKind::KIND_ADDOP)
                {
                    const Add *add = static_cast<const Add *>(operand2->getExpression());
                    const Expression *addop1 = static_cast<const Expression *>(add->getOperand1());
                    const Expression *addop2 = static_cast<const Expression *>(add->getOperand2());

                    // Start from here tomomrow. You're in the Min(x, >> Here <<); 2nd operand.
                }
                else
                {
                    llvm::outs() << "Error: Not implemented @ Tuple Assignment.\n";
                    return;
                }
            }
            else
            {
                llvm::outs() << rhsMethodCall->getIdentifier()->getname() << " Not implemented yet.\n";
                return;
            }

        }
        else
        {
            llvm::outs() << "Error: Tuple Assignment failed.\n";
            return;
        }

        if (rhsexpr2->getKind() == ExpressionKind::KIND_KEYWORD)
        {
            llvm::outs() << "Inside Keyword\n";
        }
    }

    virtual void visitAdd(const Add *add, mlir::SymbolTable *symbolTable)
    {
    }

    virtual void visitFunction(const Function *function, mlir::SymbolTable *symbolTable) override
    {
        // Create function type.
        llvm::SmallVector<mlir::Type> argTypes;
        llvm::SmallVector<mlir::Attribute> argNames;

        std::vector<mlir::Operation *> ops;

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
                    ops.push_back(declareArg.getOperation());
                    symbolTable->insert(declareArg);
                }

                else if (std::string(arg->getType()->getType()) == "Node")
                {
                    argTypes.push_back(builder.getType<mlir::starplat::NodeType>());
                    auto NodeType = mlir::starplat::NodeType::get(builder.getContext());
                    auto declareArg = builder.create<mlir::starplat::DeclareOp>(builder.getUnknownLoc(), builder.getI32Type(), ::mlir::TypeAttr::get(NodeType), builder.getStringAttr(arg->getVarName()->getname()));
                    ops.push_back(declareArg.getOperation());
                    symbolTable->insert(declareArg);
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
                    ops.push_back(declareArg.getOperation());
                    symbolTable->insert(declareArg);
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
        mlir::SymbolTable funcSymbolTable(func);

        for (auto op : ops)
            funcSymbolTable.insert(op->clone());

        stmtlist->Accept(this, &funcSymbolTable);

        // Create end operation.
        auto end = builder.create<mlir::starplat::endOp>(builder.getUnknownLoc());
    }

    virtual void visitParamlist(const Paramlist *paramlist, mlir::SymbolTable *symbolTable) override
    {
        vector<Param *> paramListVecvtor = paramlist->getParamList();

        for (Param *param : paramListVecvtor)
            param->Accept(this, symbolTable);
    }

    virtual void visitFixedpointUntil(const FixedpointUntil *fixedpointuntil, mlir::SymbolTable *symbolTable) override
    {
        // Create new region.
        // Create new symbol table.
        // Pass everything in symbol table to new symbol table.
        // Then Generate Code.

        Identifier *identifier = static_cast<Identifier *>(fixedpointuntil->getidentifier());
        Expression *expr = static_cast<Expression *>(fixedpointuntil->getexpr());
        Statementlist *stmtlist = static_cast<Statementlist *>(fixedpointuntil->getstmtlist());

        if (!symbolTable->lookup(identifier->getname()))
        {
            llvm::errs() << "Error: " << identifier->getname() << " not declared.\n";
            return;
        }

        mlir::StringAttr opAttr;
        const BoolExpr *boolExpr;
        if (expr->getKind() == ExpressionKind::KIND_BOOLEXPR)
        {
            boolExpr = static_cast<const BoolExpr *>(expr->getExpression());

            if (strcmp(boolExpr->getop(), "!") == 0)
                opAttr = builder.getStringAttr("NOT");
        }

        mlir::ArrayAttr condAttrArray = builder.getArrayAttr({opAttr});
        mlir::Operation *lhs = symbolTable->lookup(identifier->getname());
        if (!lhs)
        {
            llvm::errs() << "Error: " << identifier->getname() << " not declared.\n";
            return;
        }

        const Expression *innerBoolExpr = static_cast<const Expression *>(boolExpr->getExpr1());
        mlir::Operation *rhs;
        if (innerBoolExpr->getKind() == ExpressionKind::KIND_IDENTIFIER)
        {
            rhs = symbolTable->lookup(static_cast<const Identifier *>(innerBoolExpr->getExpression())->getname());
            if (!rhs)
            {
                llvm::errs() << "Error: " << static_cast<const Identifier *>(innerBoolExpr->getExpression())->getname() << " not declared.\n";
                return;
            }
        }

        llvm::SmallVector<mlir::Value> condArgs = {lhs->getResult(0), rhs->getResult(0)};
        auto fixedPointUntil = builder.create<mlir::starplat::FixedPointUntilOp>(builder.getUnknownLoc(), condArgs, condAttrArray);

        // Change block.
        auto &loopBlock = fixedPointUntil.getBody().emplaceBlock();
        builder.setInsertionPointToStart(&loopBlock);

        stmtlist->Accept(this, symbolTable);

        builder.create<mlir::starplat::endOp>(builder.getUnknownLoc());
        builder.setInsertionPointAfter(fixedPointUntil);
    }

    virtual void visitInitialiseAssignmentStmt(const InitialiseAssignmentStmt *initialiseAssignmentStmt, mlir::SymbolTable *symbolTable)
    {
        const TypeExpr *type = static_cast<const TypeExpr *>(initialiseAssignmentStmt->gettype());
        const Identifier *identifier = static_cast<const Identifier *>(initialiseAssignmentStmt->getidentifier());
        const Expression *expr = static_cast<const Expression *>(initialiseAssignmentStmt->getexpr());

        mlir::Type typeAttr;

        if (strcmp(type->getType(), "int") == 0)
            typeAttr = builder.getI32Type();
        else if (strcmp(type->getType(), "edge") == 0)
            typeAttr = mlir::starplat::EdgeType::get(builder.getContext());

        auto idDecl = builder.create<mlir::starplat::DeclareOp>(builder.getUnknownLoc(), builder.getI32Type(), typeAttr, builder.getStringAttr(identifier->getname()));
        symbolTable->insert(idDecl);

        expr->Accept(this, symbolTable);

        mlir::Operation *op;
        if (expr->getKind() == ExpressionKind::KIND_KEYWORD)
        {
            const Keyword *keyword = static_cast<const Keyword *>(expr->getExpression());
            if (symbolTable->lookup(keyword->getKeyword()))
                op = symbolTable->lookup(keyword->getKeyword());

            auto asgOp = builder.create<mlir::starplat::AssignmentOp>(builder.getUnknownLoc(), idDecl.getResult(), op->getResult(0));
        }

        else if (expr->getKind() == ExpressionKind::KIND_MEMBERACCESS)
        {
            const Memberaccess *memberAccessIn = static_cast<const Memberaccess *>(expr->getExpression());
            const Identifier *identifierIn = static_cast<const Identifier *>(memberAccessIn->getIdentifier());
            const Methodcall *methodcallIn = static_cast<const Methodcall *>(memberAccessIn->getMethodCall());

            if (!symbolTable->lookup(identifierIn->getname()))
            {
                llvm::outs() << "Error: Undefined variable " << identifierIn->getname() << "\n";
                return;
            }
            auto accessIdentifier = symbolTable->lookup(identifierIn->getname());

            if (methodcallIn->getIsBuiltin())
            {

                if (strcmp(methodcallIn->getIdentifier()->getname(), "get_edge") == 0)
                {
                    // Visit ParamList
                    const Paramlist *paramlist = static_cast<const Paramlist *>(methodcallIn->getParamLists());
                    paramlist->Accept(this, symbolTable);

                    vector<Param *> paramlistvector = paramlist->getParamList();
                    const Identifier *node1 = static_cast<const Identifier *>(paramlistvector[0]->getExpr()->getExpression());
                    const Identifier *node2 = static_cast<const Identifier *>(paramlistvector[1]->getExpr()->getExpression());

                    auto node1Op = symbolTable->lookup(node1->getname());
                    auto node2Op = symbolTable->lookup(node2->getname());

                    // Create a get_edge Op.
                    auto getedgeOp = builder.create<mlir::starplat::GetEdgeOp>(builder.getUnknownLoc(), typeAttr, accessIdentifier->getResult(0), node1Op->getResult(0), node2Op->getResult(0));
                }
            }
        }
    }

    virtual void visitMemberAccessAssignment(const MemberAccessAssignment *memberAccessAssignment, mlir::SymbolTable *symbolTable)
    {
        const Memberaccess *memberAccess = static_cast<const Memberaccess *>(memberAccessAssignment->getMemberAccess());
        memberAccess->Accept(this, symbolTable);

        const Expression *expr = static_cast<const Expression *>(memberAccessAssignment->getExpr());
        expr->Accept(this, symbolTable);

        const Identifier *identifier = memberAccess->getIdentifier();
        const Identifier *identifier2 = memberAccess->getIdentifier2();

        auto id1 = symbolTable->lookup(identifier->getname());
        auto id2 = symbolTable->lookup(identifier2->getname());

        if (!id1)
        {
            llvm::errs() << "Error: " << identifier->getname() << " not declared.\n";
            return;
        }

        if (!id2)
        {
            llvm::errs() << "Error: " << identifier2->getname() << " not declared.\n";
            return;
        }

        auto typeAttr = id1->getAttrOfType<mlir::TypeAttr>("type");
        mlir::Type type = typeAttr.getValue();

        if (type.isa<mlir::starplat::NodeType>())
        {
            // Set Node Property
            if (expr->getKind() == ExpressionKind::KIND_NUMBER)
            {
                const Number *number = static_cast<const Number *>(expr->getExpression());
                auto numberVal = symbolTable->lookup(std::to_string(number->getnumber()));
                auto setNodeProp = builder.create<mlir::starplat::SetNodePropertyOp>(builder.getUnknownLoc(), id1->getResult(0), id2->getResult(0), numberVal->getResult(0));
            }

            else if (expr->getKind() == ExpressionKind::KIND_KEYWORD)
            {
                const Keyword *keyword = static_cast<const Keyword *>(expr->getExpression());
                auto keywordVal = symbolTable->lookup(keyword->getKeyword());
                auto setNodeProp = builder.create<mlir::starplat::SetNodePropertyOp>(builder.getUnknownLoc(), id1->getResult(0), id2->getResult(0), keywordVal->getResult(0));
            }
        }
    }

    virtual void visitKeyword(const Keyword *keyword, mlir::SymbolTable *symbolTable) override
    {

        if (symbolTable->lookup(keyword->getKeyword()))
            return;

        auto keywrodSSA = builder.create<mlir::starplat::ConstOp>(builder.getUnknownLoc(), builder.getI32Type(), builder.getStringAttr(keyword->getKeyword()), builder.getStringAttr(keyword->getKeyword()));
        symbolTable->insert(keywrodSSA);
    }

    virtual void visitGraphProperties(const GraphProperties *graphproperties, mlir::SymbolTable *symbolTable) override
    {
    }

    virtual void visitMethodcall(const Methodcall *methodcall, mlir::SymbolTable *symbolTable) override
    {
        const Paramlist *paramlist = static_cast<const Paramlist *>(methodcall->getParamLists());
        paramlist->Accept(this, symbolTable);
    }

    virtual void visitMemberaccess(const Memberaccess *memberaccess, mlir::SymbolTable *symbolTable) override
    {

        if (!symbolTable->lookup(builder.getStringAttr(memberaccess->getIdentifier()->getname())))
        {
            llvm::outs() << "Error1: " << memberaccess->getIdentifier()->getname() << " not defined!\n";
            exit(0);
        }

        if (memberaccess->getIdentifier2())
        {
            if (!symbolTable->lookup(builder.getStringAttr(memberaccess->getIdentifier2()->getname())))
            {
                llvm::outs() << "Error2: " << memberaccess->getIdentifier2()->getname() << " not defined!\n";
                exit(0);
            }
        }
    }

    virtual void visitArglist(const Arglist *arglist, mlir::SymbolTable *symbolTable) override
    {
    }

    virtual void visitArg(const Arg *arg, mlir::SymbolTable *symbolTable) override
    {
    }

    virtual void visitStatement(const Statement *statement, mlir::SymbolTable *symbolTable) override
    {
    }

    virtual void visitStatementlist(const Statementlist *stmtlist, mlir::SymbolTable *symbolTable) override
    {
        for (ASTNode *stmt : stmtlist->getStatementList())
        {
            stmt->Accept(this, symbolTable);
        }
    }

    virtual void visitType(const TypeExpr *type, mlir::SymbolTable *symbolTable) override
    {
    }

    virtual void visitNumber(const Number *number, mlir::SymbolTable *symbolTable) override
    {
        // Create constant operation.
        if (symbolTable->lookup(std::to_string(number->getnumber())))
            return;

        auto constant = builder.create<mlir::starplat::ConstOp>(builder.getUnknownLoc(), builder.getI32Type(), builder.getStringAttr(std::to_string(number->getnumber())), builder.getStringAttr(std::to_string(number->getnumber())));
        symbolTable->insert(constant);
    }

    virtual void visitExpression(const Expression *expr, mlir::SymbolTable *symbolTable) override
    {
        expr->getExpression()->Accept(this, symbolTable);
    }

    void print()
    {
        verify(module);
        module.dump();
    }

    mlir::SymbolTable *getSymbolTable()
    {
        return &globalSymbolTable;
    }

private:
    mlir::MLIRContext context;
    mlir::OpBuilder builder;
    mlir::ModuleOp module;
    mlir::SymbolTable globalSymbolTable;
};
