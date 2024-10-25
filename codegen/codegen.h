#include "../ast/visitor.h"
#include <map>
#include <string>
#include <iostream>

using namespace std;

class CodeGen : public Visitor
{

public:
    CodeGen() : result(0) {}

    virtual void visitDeclarationStmt(const DeclarationStatement *dclstmt) override
    {
        std::cout << "Visit Decl Statement\n";
    }

    virtual void visitForallStmt(const ForallStatement *forAllStmt) override
    {
        std::cout << "Visit Foralll Statement\n";
    }

    virtual void visitIfStmt(const IfStatement *ifStmt) override
    {
        std::cout << "Visit If Statement\n";
    }

    virtual void visitBoolExpr(const BoolExpr *boolExpr) override
    {
        std::cout << "Visit Bool Statement\n";
    }

    virtual void visitIncandassignstmt(const Incandassignstmt *incandassignstmt) override
    {
        std::cout << "Visit Inc and Assign Statement\n";
    }

    virtual void visitIdentifier(const Identifier *identifier) override
    {
        std::cout << "Visit Identifier Statement\n";
    }

    virtual void visitReturnStmt(const ReturnStmt *returnStmt) override
    {
        std::cout << "Visit Return Statement\n";
    }

    virtual void visitFunction(const Function *function) override
    {
            Arglist* arglist = dynamic_cast<Arglist*> (function->getparams());
            Identifier* funcName = dynamic_cast<Identifier*> (function->getfuncname());
            Statementlist* stmtlist =  dynamic_cast<Statementlist*> (function->getstmtlist());

            cout<<"Hello from "<<funcName->getname()<<"\n";
            cout<<"Statement List "<<stmtlist->getStatementList().size();
    }

    virtual void visitParamlist(const Paramlist *paramlist) override
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

    virtual void visitStatementlist(const Statementlist *arglist) override
    {

    }





private:
    int result;
    map<string, int> variables;
};