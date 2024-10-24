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
        std::cout << "Visit Function\n";
    }

private:
    int result;
    map<string, int> variables;
};