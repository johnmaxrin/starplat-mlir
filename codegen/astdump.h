#include "../ast/visitor.h"
#include "../ast/ast.h"
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
        
        Type* type = static_cast<Type*>(dclstmt->gettype());
        Identifier* identifier = static_cast<Identifier*>(dclstmt->getvarname());
        Number* number = static_cast<Number*>(dclstmt->getnumber());

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
        std::cout << "Forall: {\n";
        
        std::cout << "Loop Var: {\n\t";
        forAllStmt->getLoopVar()->Accept(this);
        std::cout << "\t}\n";

        std::cout << "Loop Expr: {\n\t";
        forAllStmt->getexpr()->Accept(this);
        std::cout <<"}\n";

        std::cout << "Loop Body: {\n\t";
        forAllStmt->getblcstmt()->Accept(this);
        std::cout <<"}\n";

        cout<<"}\n";
    }

    virtual void visitIfStmt(const IfStatement *ifStmt) override
    {
        std::cout << "If Statement: {\n";
        ifStmt->getexpr()->Accept(this);
        ifStmt->getstmt()->Accept(this);
        std::cout << "}\n";
    }

    virtual void visitBoolExpr(const BoolExpr *boolExpr) override
    {
        std::cout << "Visit Bool Statement\n";
    }

    virtual void visitIncandassignstmt(const Incandassignstmt *incandassignstmt) override
    {
        std::cout << "Increment and Assign Statement: {\n";
        incandassignstmt->getIdentifier()->Accept(this);
        incandassignstmt->getexpr()->Accept(this);

        cout<<"}\n";
    }

    virtual void visitIdentifier(const Identifier *identifier) override
    {
        cout<<"Identifier: "<<identifier->getname()<<"\n";
    }

    virtual void visitReturnStmt(const ReturnStmt *returnStmt) override
    {
        ASTNode* expr = returnStmt->getexpr();
        
        cout<<"Return: {\n";
        expr->Accept(this);
        cout<<"}";
    }

    virtual void visitFunction(const Function *function) override
    {
            Arglist* arglist = static_cast<Arglist*> (function->getparams());
            Identifier* funcName = static_cast<Identifier*> (function->getfuncname());
            Statementlist* stmtlist =  static_cast<Statementlist*> (function->getstmtlist());

            cout<<"Function: "<<funcName->getname()<<" {\n";
            arglist->Accept(this);

            cout<<"Function Body: {\n";
            stmtlist->Accept(this);
            cout<<"\n}\n";

            
            cout<<"\n}";
    }

    virtual void visitParamlist(const Paramlist *paramlist) override
    {

    }

    virtual void visitMethodcall(const Methodcall *methodcall) override
    {       
            cout<<"Methodcall: {\n\t";
            methodcall->getIdentifier()->Accept(this);
            
            cout<<"\t";
            if(methodcall->getnode() != nullptr)
                methodcall->getnode()->Accept(this);
            cout<<"}\n";
    }

    virtual void visitMemberaccess(const Memberaccess *memberaccess) override
    {
        cout << "Member Access: {\n\t";
        memberaccess->getIdentifier()->Accept(this);

        cout<<"\t";
        if(memberaccess->getnode() != nullptr)
            memberaccess->getnode()->Accept(this);
        
        cout<<"}\n";
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
            
    }  





private:
    int result;
    map<string, int> variables;
};