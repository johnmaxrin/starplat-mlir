#include "astdump.h"
#include <iostream>

using namespace std;

CodeGen::CodeGen() : result(0) {}

void CodeGen::visitDeclarationStmt(const DeclarationStatement* dclstmt) {
    cout << "Declaration Statememt: {\n";

    TypeExpr* type         = static_cast<TypeExpr*>(dclstmt->gettype());
    Identifier* identifier = static_cast<Identifier*>(dclstmt->getvarname());
    Number* number         = static_cast<Number*>(dclstmt->getnumber());

    cout << "\t";
    type->Accept(this);

    cout << "\t";
    identifier->Accept(this);

    cout << "\t";
    number->Accept(this);

    cout << "}\n";
}

void CodeGen::visitAdd(const Add* add) {
    llvm::errs() << "visitAdd not implemented\n";
    std::abort();
}

void CodeGen::visitMemberaccessStmt(const MemberacceessStmt* memberAccess) {
    memberAccess->getMemberAccess()->Accept(this);
    // std::abort();
}

void CodeGen::visitTemplateDeclarationStmt(const TemplateDeclarationStatement* templateDeclStmt) { cout << "Template Declaration\n\n"; }

void CodeGen::visitTemplateType(const TemplateType* templateType) { cout << "Template Type\n"; }

void CodeGen::visitForallStmt(const ForallStatement* forAllStmt) {
    std::cout << "Forall: {\n";

    std::cout << "Loop Var: {\n\t";
    forAllStmt->getLoopVar()->Accept(this);
    std::cout << "\t}\n";

    std::cout << "Loop Expr: {\n\t";
    forAllStmt->getexpr()->Accept(this);
    std::cout << "}\n";

    std::cout << "Loop Body: {\n\t";
    // forAllStmt->getblcstmt()->Accept(this);
    std::cout << "}\n";

    cout << "}\n";
}

void CodeGen::visitIfStmt(const IfStatement* ifStmt) {
    std::cout << "If Statement: {\n";
    ifStmt->getexpr()->Accept(this);
    ifStmt->getstmt()->Accept(this);
    std::cout << "}\n";
}

void CodeGen::visitBoolExpr(const BoolExpr* boolExpr) { std::cout << "Visit Bool Statement\n"; }

void CodeGen::visitIncandassignstmt(const Incandassignstmt* incandassignstmt) {
    std::cout << "Increment and Assign Statement: {\n";
    // incandassignstmt->getIdentifier()->Accept(this);
    // incandassignstmt->getexpr()->Accept(this);

    cout << "}\n";
}

void CodeGen::visitAssignment(const Assignment* assignment) {}

void CodeGen::visitAssignmentStmt(const AssignmentStmt* assignemntStmt) {}

void CodeGen::visitIdentifier(const Identifier* identifier) { cout << "Identifier: " << identifier->getname() << "\n"; }

void CodeGen::visitReturnStmt(const ReturnStmt* returnStmt) {
    ASTNode* expr = returnStmt->getexpr();

    cout << "Return: {\n";
    expr->Accept(this);
    cout << "}";
}

void CodeGen::visitParameterAssignment(const ParameterAssignment* paramAssignment) { cout << "Parameter Assignment\n"; }

void CodeGen::visitParam(const Param* param) { cout << "Params\n\n"; }

void CodeGen::visitTupleAssignment(const TupleAssignment* tupleAssignment) { cout << "Tuple Assignment\n"; }

void CodeGen::visitFunction(const Function* function) {
    Arglist* arglist        = static_cast<Arglist*>(function->getparams());
    Identifier* funcName    = static_cast<Identifier*>(function->getfuncname());
    Statementlist* stmtlist = static_cast<Statementlist*>(function->getstmtlist());

    cout << "Function: " << funcName->getname() << " {\n";
    arglist->Accept(this);

    cout << "Function Body: {\n";
    stmtlist->Accept(this);
    cout << "\n}\n";

    cout << "\n}";
}

void CodeGen::visitParamlist(const Paramlist* paramlist) { cout << "ParamList\n\n"; }

void CodeGen::visitFixedpointUntil(const FixedpointUntil* fixedpointuntil) {
    Statementlist* stmtlist = static_cast<Statementlist*>(fixedpointuntil->getstmtlist());
    cout << "Fixed Point Until\n";
    stmtlist->Accept(this);
}

void CodeGen::visitInitialiseAssignmentStmt(const InitialiseAssignmentStmt* initialiseAssignmentStmt) { cout << "Init Assignment Stmt\n"; }

void CodeGen::visitMemberAccessAssignment(const MemberAccessAssignment* memberAccessAssignment) { cout << "Member Access Assignment\n\n"; }

void CodeGen::visitKeyword(const Keyword* keyword) {}

void CodeGen::visitGraphProperties(const GraphProperties* graphproperties) { cout << graphproperties->getPropertyType() << " "; }

void CodeGen::visitMethodcall(const Methodcall* methodcall) {
    cout << "Methodcall: {\n\t";
    methodcall->getIdentifier()->Accept(this);

    cout << "\t";
    if (methodcall->getParamLists() != nullptr)
        methodcall->getParamLists()->Accept(this);
    cout << "}\n";
}

void CodeGen::visitMemberaccess(const Memberaccess* memberaccess) {
    cout << "Member Access: {\n\t";
    memberaccess->getIdentifier()->Accept(this);

    cout << "\t";
    if (memberaccess->getMethodCall() != nullptr)
        memberaccess->getMethodCall()->Accept(this);

    cout << "}\n";
}

void CodeGen::visitArglist(const Arglist* arglist) {
    vector<Arg*> arglistV = arglist->getArgList();

    cout << "Arguments: {\n";

    for (Arg* arg : arglistV) {
        cout << "Arg: {\n";

        arg->Accept(this);

        cout << "   }\n";
    }

    cout << "}\n";
}

void CodeGen::visitArg(const Arg* arg) {
    if (arg->getType()) {
        cout << "\t";
        arg->getType()->Accept(this);

        cout << "\t";
        arg->getVarName()->Accept(this);
    }

    else if (arg->getTemplateType()) {
        cout << "\t{";
        cout << "PROPxx";
        cout << "}";
    }
}

void CodeGen::visitStatement(const Statement* statement) {}

void CodeGen::visitStatementlist(const Statementlist* stmtlist) {
    for (ASTNode* stmt : stmtlist->getStatementList()) {
        stmt->Accept(this);
    }
}

void CodeGen::visitType(const TypeExpr* type) { cout << "type: " << type->getType() << "\n"; }

void CodeGen::visitNumber(const Number* number) { cout << "number: " << number->getnumber() << "\n"; }

void CodeGen::visitExpression(const Expression* expr) {}
