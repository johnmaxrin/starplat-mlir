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
// 12. For now creating a new operation for argument. Later make use og block arguments.

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

#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

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
        symbolTables.push_back(&globalSymbolTable);
    }

    virtual void visitDeclarationStmt(const DeclarationStatement *dclstmt, mlir::SymbolTable *symbolTable) override
    {
    }

    virtual void visitTemplateDeclarationStmt(const TemplateDeclarationStatement *templateDeclStmt, mlir::SymbolTable *symbolTable) override
    {

        TemplateType *Type = static_cast<TemplateType *>(templateDeclStmt->gettype());
        Identifier *identifier = static_cast<Identifier *>(templateDeclStmt->getvarname());

        const Identifier *graphName = Type->getGraphName();
        llvm::StringRef graphId = graphName->getname();

        // TODO: Change the function name to getGraphProp().
        if (std::string(Type->getGraphPropNode()->getPropertyType()) == "propNode")
        {

            mlir::Type type;

            if(std::string(Type->getType()->getType()) == "int")
                type = builder.getType<mlir::starplat::PropNodeType>(builder.getI64Type(), graphId);
            else if(std::string(Type->getType()->getType()) == "bool")
                type = builder.getType<mlir::starplat::PropNodeType>(builder.getI1Type(), graphId);
            else
            {
                llvm::errs() <<"Error: Type not implemented\n";
                exit(0);
            } 

            auto visibility = builder.getStringAttr("public");
            mlir::Value graph = globalLookupOp(graphId);

            mlir::Operation *declareOp;

            if(graph)
                declareOp = builder.create<mlir::starplat::DeclareOp>(builder.getUnknownLoc(), type , builder.getStringAttr(identifier->getname()), visibility, graph);
            else
                llvm::errs() <<"Error: Undefined Symbol "<<graphId<<"\n";

            if (globalLookupOp(identifier->getname()))
            {
            }
            else
                symbolTable->insert(declareOp);
        }

        else if (std::string(Type->getGraphPropNode()->getPropertyType()) == "propEdge")
        {

            auto type = builder.getType<mlir::starplat::PropEdgeType>(builder.getI32Type(), graphId);
            mlir::Value graph = NULL;
            auto declare = builder.create<mlir::starplat::DeclareOp>(builder.getUnknownLoc(), type, builder.getStringAttr(identifier->getname()), builder.getStringAttr("public"), graph);
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
        std::vector<mlir::Operation *> ops; // For new SymbolTable

        auto loopVarSymbol = globalLookupOp(loopVar->getname());
        mlir::Type loopVarType;
        mlir::Operation *loopVarOp;
        mlir::SmallVector<mlir::Attribute> loopAttr;
        mlir::SmallVector<mlir::Value> loopOperands;
        mlir::BoolAttr filter = builder.getBoolAttr(0);

        if (loopVarSymbol)
        {
            llvm::outs() << "Error: Identifier '" << loopVar->getname() << "' already in declared.\n";
            exit(0);
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
                auto idSymbol = globalLookupOp(identifier->getname());

                if (!idSymbol)
                {
                    llvm::outs() << "Error: Identifier '" << identifier->getname() << "' not declared.\n";
                    exit(0);
                }

                loopOperands.push_back(idSymbol);

                const Identifier *innerMethodcallIdentifier = innerMethodcall->getIdentifier();

                if (strcmp(innerMethodcallIdentifier->getname(), "nodes") == 0)
                {
                    loopVarType = mlir::starplat::NodeType::get(builder.getContext());
                    mlir::Value graph = NULL;
                    loopVarOp = builder.create<mlir::starplat::DeclareOp>(builder.getUnknownLoc(), loopVarType, builder.getStringAttr(loopVar->getname()), builder.getStringAttr("public"), graph);
                    ops.push_back(loopVarOp);
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
                        const Expression *idExpr = static_cast<const Expression *>(methodcallin->getParamLists());

                        if (idExpr->getKind() == ExpressionKind::KIND_IDENTIFIER)
                        {
                            const Identifier *idParam = static_cast<const Identifier *>(idExpr->getExpression());
                            llvm::outs() << "Symbol: " << idParam->getname() << "\n";

                            auto idSymbol = globalLookupOp(idParam->getname());

                            if (idSymbol)
                                loopOperands.push_back(idSymbol);

                            else
                            {
                                llvm::errs() << "Symbol not found at neighnours.\n";
                                exit(0);
                            }
                        }

                        else
                        {
                            llvm::errs() << "Error: at Neighbours\n";
                            exit(0);
                        }

                        loopVarType = mlir::starplat::NodeType::get(builder.getContext());
                        mlir::Value graph = NULL;
                        loopVarOp = builder.create<mlir::starplat::DeclareOp>(builder.getUnknownLoc(), loopVarType, builder.getStringAttr(loopVar->getname()), builder.getStringAttr("public"), graph);
                        ops.push_back(loopVarOp);
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

                    // loopAttr.push_back(builder.getStringAttr("filter"));
                    filter = builder.getBoolAttr(1);

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

                        auto lhsidSymbol = globalLookupOp(lhsIdentifier->getname());
                        auto rhsKeywordSymbol = globalLookupOp(rhsKeyword->getKeyword());

                        if(!rhsKeywordSymbol)
                            rhsKeyword->Accept(this, symbolTable);

                        if (!lhsidSymbol)
                        {
                            llvm::outs() << "Error DXB: Identifier '" << lhsIdentifier->getname()<< "' not declared.\n";
                            return;
                        }

                        loopOperands.push_back(lhsidSymbol);
                        loopOperands.push_back(globalLookupOp(rhsKeyword->getKeyword()));
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
        mlir::StringAttr loopa = builder.getStringAttr("loopa");

        auto loopOp = builder.create<mlir::starplat::ForAllOp>(builder.getUnknownLoc(), loopOperands, loopAttrArray, filter, loopa);

        loopOp.setNested();

        auto &loopBlock = loopOp.getBody().emplaceBlock();

        /*
        // Visit the function body.
                mlir::SymbolTable funcSymbolTable(func);

                for (auto op : ops)
                    funcSymbolTable.insert(op->clone());

                stmtlist->Accept(this, &funcSymbolTable);

        */

        // Copy Everything from Symbol table to forAllSymbolTable.

        builder.setInsertionPointToStart(&loopBlock);

        mlir::SymbolTable forAllSymbolTable(loopOp);
        symbolTables.push_back(&forAllSymbolTable);

        stmtlist->Accept(this, &forAllSymbolTable);

        builder.create<mlir::starplat::endOp>(builder.getUnknownLoc());
        builder.setInsertionPointAfter(loopOp);
    }

    virtual void visitMemberaccessStmt(const MemberacceessStmt *MemberacceessStmt, mlir::SymbolTable *symbolTable) override
    {
        const Memberaccess *memberaccessnode = static_cast<const Memberaccess *>(MemberacceessStmt->getMemberAccess());
        const Methodcall *methodcallnode = static_cast<const Methodcall *>(memberaccessnode->getMethodCall());
        const Paramlist *paramlist = static_cast<const Paramlist *>(methodcallnode->getParamLists());

        methodcallnode->Accept(this, symbolTable);
        std::vector<Param *> paramListVecvtor = paramlist->getParamList();
        llvm::SmallVector<mlir::Value> operandsForAttachNodeProperty;

        const Identifier *identifier1 = memberaccessnode->getIdentifier();
        auto graph = globalLookupOp(identifier1->getname());
        if (!graph)
            llvm::outs() << "Error: Graph not defined!\n";
        operandsForAttachNodeProperty.push_back(graph);

        const Identifier *identifier2 = memberaccessnode->getIdentifier2();

        if (methodcallnode && methodcallnode->getIsBuiltin())
        {
            if (std::strcmp(methodcallnode->getIdentifier()->getname(), "attachNodeProperty") == 0)
            {
                // Create attachNodeProperty operation.

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

                    auto idSymbol = globalLookupOp(identifier->getname()); // globalLookupOp(identifier->getname());
                    auto kwSymbol = globalLookupOp(keyword->getKeyword()); // globalLookupOp(keyword->getKeyword());


                    if (!kwSymbol)
                        llvm::outs() << "Error: Keyword '" << keyword->getKeyword() << "' not declared.\n";

                    if (!idSymbol)
                        llvm::outs() << "Error: Identifier '" << identifier->getname() << "' not declared.\n";

                    if (idSymbol && kwSymbol)
                    {
                        operandsForAttachNodeProperty.push_back(idSymbol);
                        operandsForAttachNodeProperty.push_back(kwSymbol);
                    }
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
        keyword->Accept(this, symbolTable);


        if (globalLookupOp(identifier->getname()))
        {
            auto lhs = globalLookupOp(identifier->getname());
            auto rhs = globalLookupOp(keyword->getKeyword());

            // auto assign = builder.create<mlir::starplat::AssignmentOp>(builder.getUnknownLoc(), lhs->getResult(0), rhs->getResult(0));
            // symbolTable->rename(assign, builder.getStringAttr(identifier->getname()));
        }

        else if (globalLookupOp(identifier->getname()) != nullptr)
        {

            auto lhs = globalLookupOp(identifier->getname());
            auto rhs = globalLookupOp(keyword->getKeyword());

            // auto assign = builder.create<mlir::starplat::AssignmentOp>(builder.getUnknownLoc(), lhs->getResult(0), rhs->getResult(0));
            // symbolTable->rename(assign, builder.getStringAttr(identifier->getname()));
        }
        else
        {
            llvm::outs() << "error: " << identifier->getname() << " not declared -_-\n";
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

        mlir::Operation *gOperand1;
        mlir::Operation *gOperand2;
        mlir::Operation *gOperand3;
        mlir::Value gOperand4;

        if (lhsexpr1->getKind() == ExpressionKind::KIND_MEMBERACCESS)
        {
            const Memberaccess *lhs1MemberAccess = static_cast<const Memberaccess *>(lhsexpr1->getExpression());
            if (lhs1MemberAccess->getIdentifier() && lhs1MemberAccess->getIdentifier2())
            {
                if (globalLookupOp(lhs1MemberAccess->getIdentifier2()->getname()))
                {
                    if (globalLookupOp(lhs1MemberAccess->getIdentifier()->getname()))
                    {
                        mlir::Value propOp = globalLookupOp(lhs1MemberAccess->getIdentifier2()->getname());
                        mlir::Value varOp = globalLookupOp(lhs1MemberAccess->getIdentifier()->getname());
                        gOperand1 = builder.create<mlir::starplat::GetNodePropertyOp>(builder.getUnknownLoc(), builder.getI32Type(), varOp, propOp, builder.getStringAttr(lhs1MemberAccess->getIdentifier2()->getname()));
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
                if (globalLookupOp(lhs2MemberAccess->getIdentifier2()->getname()))
                {
                    if (globalLookupOp(lhs2MemberAccess->getIdentifier()->getname()))
                    {
                        mlir::Value propOp = globalLookupOp(lhs2MemberAccess->getIdentifier2()->getname());
                        mlir::Value varOp = globalLookupOp(lhs2MemberAccess->getIdentifier()->getname());
                        gOperand1 = builder.create<mlir::starplat::GetNodePropertyOp>(builder.getUnknownLoc(), builder.getI32Type(), varOp, propOp, builder.getStringAttr(lhs2MemberAccess->getIdentifier2()->getname()));
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
                        if (!globalLookupOp(id1->getname()))
                        {
                            llvm::outs() << id1->getname() << " not defined.\n";
                            exit(0);
                        }
                        if (!globalLookupOp(id2->getname()))
                        {
                            llvm::outs() << id2->getname() << " not defined.\n";
                            exit(0);
                        }

                        auto id1Op = globalLookupOp(id1->getname());
                        auto id2Op = globalLookupOp(id2->getname());

                        auto id1Opop = id1Op.getDefiningOp();
                        auto typeAttr = id1Opop->getAttrOfType<mlir::TypeAttr>("type");
                        
                        mlir::Type id1Optype = typeAttr.getValue();

                        if (id1Optype.isa<mlir::starplat::NodeType>())
                        {
                            // Generate get node property.
                            llvm::StringRef nameRef(id2->getname());
                            gOperand2 = builder.create<mlir::starplat::GetNodePropertyOp>(builder.getUnknownLoc(), builder.getI32Type(), id1Op, id2Op, builder.getStringAttr(nameRef));
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

                    // Create an Add Op
                    mlir::Operation *op1;
                    mlir::Operation *op2;

                    if (addop1->getKind() == ExpressionKind::KIND_MEMBERACCESS)
                    {
                        const Memberaccess *op1MemberAccess = static_cast<const Memberaccess *>(addop1->getExpression());
                        const Identifier *op1Id1 = static_cast<const Identifier *>(op1MemberAccess->getIdentifier());
                        const Identifier *op1Id2 = static_cast<const Identifier *>(op1MemberAccess->getIdentifier2());

                        if (globalLookupOp(op1Id1->getname()) && globalLookupOp(op1Id2->getname()))
                        {
                            auto op1id1op = globalLookupOp(op1Id1->getname());
                            auto op1id2op = globalLookupOp(op1Id2->getname());

                            auto op1id1opop = op1id1op.getDefiningOp();

                            auto typeAttr = op1id1opop->getAttrOfType<mlir::TypeAttr>("type");
                            mlir::Type type = typeAttr.getValue();


                            if (type.isa<mlir::starplat::NodeType>())
                            {
                                // Generate getNodeProp
                                auto getProp = builder.getStringAttr(op1Id2->getname()); // TODO: Add check here
                                op1 = builder.create<mlir::starplat::GetNodePropertyOp>(builder.getUnknownLoc(), builder.getI32Type(), op1id1op, op1id2op, getProp);
                            }

                            else
                            {
                                llvm::outs() << "Error: Type error in Min Fucn";
                                exit(0);
                            }
                        }

                        else
                        {
                            llvm::outs() << "Error: Undefined variables in Add Operation in methodcall in Min funciton ;>\n";
                            exit(0);
                        }
                    }
                    else
                    {
                        llvm::outs() << "Error: Add Op Not Implemented.\n";
                        exit(0);
                    }

                    // // ---- //
                    if (addop2->getKind() == ExpressionKind::KIND_MEMBERACCESS)
                    {
                        const Memberaccess *op2MemberAccess = static_cast<const Memberaccess *>(addop2->getExpression());
                        const Identifier *op2Id1 = static_cast<const Identifier *>(op2MemberAccess->getIdentifier());
                        const Identifier *op2Id2 = static_cast<const Identifier *>(op2MemberAccess->getIdentifier2());

                        if (globalLookupOp(op2Id1->getname()) && globalLookupOp(op2Id2->getname()))
                        {
                            auto op2id1op = globalLookupOp(op2Id1->getname());
                            auto op2id2op = globalLookupOp(op2Id2->getname());

                            auto op2id1opop = op2id1op.getDefiningOp();
                            auto typeAttr = op2id1opop->getAttrOfType<mlir::TypeAttr>("type");
                            mlir::Type type = typeAttr.getValue();

                            if (type.isa<mlir::starplat::EdgeType>())
                            {
                                // Generate getNodeProp
                                auto getProp = builder.getStringAttr(op2Id2->getname()); // TODO: Add check here
                                op2 = builder.create<mlir::starplat::GetEdgePropertyOp>(builder.getUnknownLoc(), builder.getI32Type(), op2id1op, op2id2op, getProp);
                            }

                            else
                            {
                                llvm::outs() << "Error: Type error in Min Fucn";
                                exit(0);
                            }
                        }

                        else
                        {
                            llvm::outs() << "Error: Undefined variables " << op2Id1->getname() << " or " << op2Id2->getname() << "\n";
                            exit(0);
                        }
                    }
                    else
                    {
                        llvm::outs() << "Error: Add Op Not Implemented.\n";
                        exit(0);
                    }

                    // Create add OP
                    gOperand3 = builder.create<mlir::starplat::AddOp>(builder.getUnknownLoc(), builder.getI32Type(), op1->getResult(0), op2->getResult(0));
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

            if (rhsexpr2->getKind() == ExpressionKind::KIND_KEYWORD)
            {
                const Keyword *rhsKeyword = static_cast<const Keyword *>(rhsexpr2->getExpression());
                if (globalLookupOp(rhsKeyword->getKeyword()))
                {
                    gOperand4 = globalLookupOp(rhsKeyword->getKeyword());
                }
                else
                {
                    llvm::outs() << "Error: Add this to Symbol Table and Pass the operation.\n";
                    exit(0);
                }
            }

            // Create the Min Tupple

            auto minOp = builder.create<mlir::starplat::MinOp>(builder.getUnknownLoc(), builder.getI32Type(), gOperand2->getResult(0), gOperand1->getResult(0), gOperand2->getResult(0), gOperand3->getResult(0), gOperand4);
        }
        else
        {
            llvm::outs() << "Error: Tuple Assignment failed.\n";
            return;
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
        llvm::SmallVector<mlir::Type> arg4Builder;

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
                }

                else if (std::string(arg->getType()->getType()) == "Node")
                {
                    argTypes.push_back(builder.getType<mlir::starplat::NodeType>());
                    auto NodeType = mlir::starplat::NodeType::get(builder.getContext());
                }
            }
            else if (arg->getTemplateType() != nullptr)
            {
                llvm::StringRef graphId = arg->getTemplateType()->getGraphName()->getname();

                if (std::string(arg->getTemplateType()->getGraphPropNode()->getPropertyType()) == "propNode")
                {
                    argTypes.push_back(builder.getType<mlir::starplat::PropNodeType>(builder.getI32Type(), graphId));
                    auto type = builder.getType<mlir::starplat::PropNodeType>(builder.getI32Type(), graphId);
                    auto typeAttr = ::mlir::TypeAttr::get(type);
                }
                else if (std::string(arg->getTemplateType()->getGraphPropNode()->getPropertyType()) == "propEdge")
                {
                    argTypes.push_back(builder.getType<mlir::starplat::PropNodeType>(builder.getI32Type(), graphId));
                    auto type = builder.getType<mlir::starplat::PropNodeType>(builder.getI32Type(), graphId);
                    auto typeAttr = ::mlir::TypeAttr::get(type);
                }
            }

            argNames.push_back(builder.getStringAttr(arg->getVarName()->getname()));
        }

        auto funcType = builder.getFunctionType(argTypes, {});
        mlir::ArrayAttr argNamesAttr = builder.getArrayAttr(argNames);

        auto func = builder.create<mlir::starplat::FuncOp>(builder.getUnknownLoc(), function->getfuncNameIdentifier(), funcType, argNamesAttr);
        //func.setNested();

        module.push_back(func);
        auto &entryBlock = func.getBody().emplaceBlock();

        int idx = 0;
        for (auto arg : funcType.getInputs())
        {
            auto argval = entryBlock.addArgument(arg, builder.getUnknownLoc());
            auto argName = argNames[idx++];
            nameToArgMap[argName.dyn_cast<mlir::StringAttr>().getValue()] = argval;

        }

        // Visit the function body.
        Statementlist *stmtlist = static_cast<Statementlist *>(function->getstmtlist());

        mlir::SymbolTable funcSymbolTable(func);
        symbolTables.push_back(&funcSymbolTable);

        builder.setInsertionPointToStart(&entryBlock);

        // for (auto argItr : args)
        // {

        //     auto argOp = builder.create<mlir::starplat::ArgOp>(builder.getUnknownLoc(), builder.getI32Type(), argTypes[idx++], builder.getStringAttr(argItr->getVarName()->getname()), builder.getStringAttr("public"));
        //     funcSymbolTable.insert(argOp);
        // }

        stmtlist->Accept(this, &funcSymbolTable);

        // Create end operation.
        auto returnOp = builder.create<mlir::starplat::ReturnOp>(builder.getUnknownLoc());
    }

    virtual void visitParamlist(const Paramlist *paramlist, mlir::SymbolTable *symbolTable) override
    {
        std::vector<Param *> paramListVecvtor = paramlist->getParamList();

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

        if (!globalLookupOp(identifier->getname()))
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

        mlir::Value lhs = globalLookupOp(identifier->getname());
        if (!lhs)
        {
            llvm::errs() << "Error: " << identifier->getname() << " not declared.\n";
            return;
        }

        const Expression *innerBoolExpr = static_cast<const Expression *>(boolExpr->getExpr1());
        mlir::Value rhs;
        if (innerBoolExpr->getKind() == ExpressionKind::KIND_IDENTIFIER)
        {
            rhs = globalLookupOp(static_cast<const Identifier *>(innerBoolExpr->getExpression())->getname());
            if (!rhs)
            {
                llvm::errs() << "Error: " << static_cast<const Identifier *>(innerBoolExpr->getExpression())->getname() << " not declared.\n";
                return;
            }
        }

        llvm::SmallVector<mlir::Value> condArgs = {lhs, rhs};
        mlir::StringAttr fixPntAttr = builder.getStringAttr("FixedPnt");
        auto fixedPointUntil = builder.create<mlir::starplat::FixedPointUntilOp>(builder.getUnknownLoc(), condArgs, condAttrArray, fixPntAttr);

        fixedPointUntil.setNested();

        // Change block.
        auto &loopBlock = fixedPointUntil.getBody().emplaceBlock();
        builder.setInsertionPointToStart(&loopBlock);

        mlir::SymbolTable fixedSymbolTable(fixedPointUntil);
        symbolTables.push_back(&fixedSymbolTable);

        stmtlist->Accept(this, &fixedSymbolTable);

        builder.create<mlir::starplat::endOp>(builder.getUnknownLoc());
        builder.setInsertionPointAfter(fixedPointUntil);
    }

    virtual void visitInitialiseAssignmentStmt(const InitialiseAssignmentStmt *initialiseAssignmentStmt, mlir::SymbolTable *symbolTable)
    {
        const TypeExpr *type = static_cast<const TypeExpr *>(initialiseAssignmentStmt->gettype());
        const Identifier *identifier = static_cast<const Identifier *>(initialiseAssignmentStmt->getidentifier());
        const Expression *expr = static_cast<const Expression *>(initialiseAssignmentStmt->getexpr());

        mlir::Type typeAttr;

        mlir::Value lhs = NULL;
        mlir::Value rhs = NULL;

        if (strcmp(type->getType(), "int") == 0)
            typeAttr = builder.getI64Type();
        
        else if (strcmp(type->getType(), "bool") == 0)
            typeAttr = builder.getI1Type();

        else if (strcmp(type->getType(), "edge") == 0)
            typeAttr = mlir::starplat::EdgeType::get(builder.getContext());

        mlir::Value graph = NULL;
        auto idDecl = builder.create<mlir::starplat::DeclareOp>(builder.getUnknownLoc(), typeAttr, builder.getStringAttr(identifier->getname()), builder.getStringAttr("public"), graph);
        lhs = idDecl.getResult();

        symbolTable->insert(idDecl);

        expr->Accept(this, symbolTable);

        mlir::Value op;
        if (expr->getKind() == ExpressionKind::KIND_KEYWORD)
        {
            const Keyword *keyword = static_cast<const Keyword *>(expr->getExpression());
            if (globalLookupOp(keyword->getKeyword()))
                op = globalLookupOp(keyword->getKeyword());

            auto asgOp = builder.create<mlir::starplat::AssignmentOp>(builder.getUnknownLoc(), idDecl.getResult(), op);
        }

        else if (expr->getKind() == ExpressionKind::KIND_MEMBERACCESS)
        {
            const Memberaccess *memberAccessIn = static_cast<const Memberaccess *>(expr->getExpression());
            const Identifier *identifierIn = static_cast<const Identifier *>(memberAccessIn->getIdentifier());
            const Methodcall *methodcallIn = static_cast<const Methodcall *>(memberAccessIn->getMethodCall());

            if (!globalLookupOp(identifierIn->getname()))
            {
                llvm::outs() << "Error: Undefined variable " << identifierIn->getname() << "\n";
                return;
            }
            auto accessIdentifier = globalLookupOp(identifierIn->getname());

            if (methodcallIn->getIsBuiltin())
            {

                if (strcmp(methodcallIn->getIdentifier()->getname(), "get_edge") == 0)
                {
                    // Visit ParamList
                    const Paramlist *paramlist = static_cast<const Paramlist *>(methodcallIn->getParamLists());
                    paramlist->Accept(this, symbolTable);

                    std::vector<Param *> paramlistvector = paramlist->getParamList();
                    const Identifier *node1 = static_cast<const Identifier *>(paramlistvector[0]->getExpr()->getExpression());
                    const Identifier *node2 = static_cast<const Identifier *>(paramlistvector[1]->getExpr()->getExpression());

                    auto node1Op = globalLookupOp(node1->getname());
                    auto node2Op = globalLookupOp(node2->getname());

                    // Create a get_edge Op.
                    auto getedgeOp = builder.create<mlir::starplat::GetEdgeOp>(builder.getUnknownLoc(), typeAttr, accessIdentifier, node1Op, node2Op);
                    rhs = getedgeOp.getResult();
                    // Assign getedgeOp to iDecl.
                    // Work here tomorrow.
                    builder.create<mlir::starplat::AssignmentOp>(builder.getUnknownLoc(), lhs, rhs);
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

        auto id1 = globalLookupOp(identifier->getname());
        auto id2 = globalLookupOp(identifier2->getname());

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
        mlir::starplat::NodeType::get(builder.getContext());
        // auto typeAttr = id1->getAttrOfType<mlir::TypeAttr>("type");
        mlir::Type type = id1.getType();


        if (type.isa<mlir::starplat::NodeType>())
        {
            // Set Node Property
            if (expr->getKind() == ExpressionKind::KIND_NUMBER)
            {
                const Number *number = static_cast<const Number *>(expr->getExpression());
                auto numberVal = globalLookupOp(std::to_string(number->getnumber()));
                auto setNodeProp = builder.create<mlir::starplat::SetNodePropertyOp>(builder.getUnknownLoc(),id2.getDefiningOp()->getOperand(0), id1, id2, numberVal);
            }

            else if (expr->getKind() == ExpressionKind::KIND_KEYWORD)
            {
                const Keyword *keyword = static_cast<const Keyword *>(expr->getExpression());
                auto keywordVal = globalLookupOp(keyword->getKeyword());
                auto setNodeProp = builder.create<mlir::starplat::SetNodePropertyOp>(builder.getUnknownLoc(),id2.getDefiningOp()->getOperand(0), id1, id2, keywordVal);
            }

        }
    }

    virtual void visitKeyword(const Keyword *keyword, mlir::SymbolTable *symbolTable) override
    {

        if (globalLookupOp(keyword->getKeyword()))
            return;

        mlir::Operation *keywordSSA;

        if(strcmp(keyword->getKeyword(),"False") == 0)
            keywordSSA = builder.create<mlir::starplat::ConstOp>(builder.getUnknownLoc(), builder.getI1Type(), builder.getStringAttr(keyword->getKeyword()), builder.getStringAttr(keyword->getKeyword()), builder.getStringAttr("public"));
        
        else if(strcmp(keyword->getKeyword(),"True") == 0)
            keywordSSA = builder.create<mlir::starplat::ConstOp>(builder.getUnknownLoc(), builder.getI1Type(), builder.getStringAttr(keyword->getKeyword()), builder.getStringAttr(keyword->getKeyword()), builder.getStringAttr("public"));
        else
            keywordSSA = builder.create<mlir::starplat::ConstOp>(builder.getUnknownLoc(), builder.getI64Type(), builder.getStringAttr(keyword->getKeyword()), builder.getStringAttr(keyword->getKeyword()), builder.getStringAttr("public"));


        symbolTable->insert(keywordSSA);
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

        if (!globalLookupOp(builder.getStringAttr(memberaccess->getIdentifier()->getname())))
        {
            if (globalLookupOp(memberaccess->getIdentifier()->getname()) == nullptr)
            {
                llvm::outs() << "Error1: " << memberaccess->getIdentifier()->getname() << " not defined!\n";
                exit(0);
            }
        }

        if (memberaccess->getIdentifier2())
        {
            if (!globalLookupOp(builder.getStringAttr(memberaccess->getIdentifier2()->getname())))
            {
                if (globalLookupOp(memberaccess->getIdentifier()->getname()) == nullptr)
                {
                    llvm::outs() << "Error2: " << memberaccess->getIdentifier2()->getname() << " not defined!\n";
                    exit(0);
                }
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
        if (globalLookupOp(std::to_string(number->getnumber())))
            return;

        auto constant = builder.create<mlir::starplat::ConstOp>(builder.getUnknownLoc(), builder.getI32Type(), builder.getStringAttr(std::to_string(number->getnumber())), builder.getStringAttr(std::to_string(number->getnumber())), builder.getStringAttr("public"));
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

    mlir::MLIRContext *getContext()
    {
        return &context;
    }

    mlir::ModuleOp *getModule()
    {
        return &module;
    }

private:
    mlir::MLIRContext context;
    mlir::OpBuilder builder;
    mlir::ModuleOp module;
    mlir::SymbolTable globalSymbolTable;

    std::vector<mlir::SymbolTable *> symbolTables;
    llvm::DenseMap<StringRef, mlir::Value> nameToArgMap;

    mlir::Value globalLookupOp(llvm::StringRef name)
    {
        auto it = nameToArgMap.find(name);
        if (it != nameToArgMap.end())
        {
            mlir::Value value = it->second;
            return value;
        }

        for (auto symbolTable : symbolTables)
        {
            if (symbolTable->lookup(name))
                return symbolTable->lookup(name)->getResult(0);
        }

        return nullptr;
    }

};
