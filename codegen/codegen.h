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
        Type* type = dynamic_cast<Type*>(dclstmt->gettype());
        cout<<"Type "<<type->getType()<<"\n";
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
        cout<<"\tname: "<<identifier->getname();
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

            cout<<"Function: "<<funcName->getname()<<" {\n";
            arglist->Accept(this);

            cout<<"Body: {\n";
            stmtlist->Accept(this);
            cout<<"\n}\n";

            
            cout<<"\n}";
    }

    virtual void visitParamlist(const Paramlist *paramlist) override
    {

    }

    virtual void visitArglist(const Arglist *arglist) override
    {
        vector<Arg*> arglistV =  arglist->getArgList();

        cout<<"Arguments: {\n";
        
        for(Arg* arg: arglistV)
        {
            cout<<"Arg: {\n\t";

            arg->Accept(this);
            
            cout<<"\n}\n";
        }
            
        

        cout<<"}\n";
    }

    virtual void visitArg(const Arg *arg) override
    {
        arg->getType()->Accept(this);
        arg->getVarName()->Accept(this);
    }

    virtual void visitStatement(const Statement *statement) override
    {
            cout<<"Hello\n";
    }

    virtual void visitStatementlist(const Statementlist *stmtlist) override
    {
            for(ASTNode *stmt: stmtlist->getStatementList())
            {
                stmt->Accept(this);
            }
    }

    virtual void visitType(const Type *type) override
    {
            cout<<"type: "<<type->getType()<<"\n";
    }





private:
    int result;
    map<string, int> variables;
};