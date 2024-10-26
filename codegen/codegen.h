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
        cout<<"Declaration Statememt: {\n";
        
        Type* type = dynamic_cast<Type*>(dclstmt->gettype());
        Identifier* identifier = dynamic_cast<Identifier*>(dclstmt->getvarname());
        Number* number = dynamic_cast<Number*>(dclstmt->getnumber());

        cout<<"\t";
        type->Accept(this);
        
        cout<<"\t";
        identifier->Accept(this);
        
        cout<<"\t";
        number->Accept(this);

        cout<<"}\n";

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
        cout<<"name: "<<identifier->getname()<<"\n";
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
            cout<<"Arg: {\n";

            arg->Accept(this);
            
            cout<<"   }\n";
        }
            
        

        cout<<"}\n";
    }

    virtual void visitArg(const Arg *arg) override
    {   
        cout<<"\t";
        arg->getType()->Accept(this);
        
        cout<<"\t";
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

    
    virtual void visitNumber(const Number *number) override
    {
            cout<<"number: "<<number->getnumber()<<"\n";
    }

    
    virtual void visitExpression(const Expression *expr) override
    {
            cout<<"expression: ";
    }  





private:
    int result;
    map<string, int> variables;
};