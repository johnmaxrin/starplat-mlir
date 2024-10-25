#ifndef AVIALAST
#define AVIALAST

#include "visitor.h"
#include <vector>
#include <iostream>

using namespace std;

class ASTNode
{
    public:
        virtual ~ASTNode() {}
        virtual void Accept(Visitor *visitor) const = 0;
};

class Statement : public ASTNode
{
    public:
        Statement(){}

        virtual void Accept(Visitor *visitor) const override{
            visitor->visitStatement(this);
        }
        
        ~Statement(){
            delete statement;
        }

        ASTNode* getstatement() const{return statement;}
    
    private:
        ASTNode *statement;
};

class Statementlist : public ASTNode
{
    public:
    
        Statementlist(){}

        virtual void Accept(Visitor *visitor) const override{
            visitor->visitStatementlist(this);
        }
        
        ~Statementlist(){
            for(ASTNode* param: statementlist)
                delete param;
        }

        const vector<ASTNode*> getStatementList() const{return statementlist;}
        void addstmt(ASTNode* param){statementlist.push_back(param);}
    
    private:
        vector<ASTNode*> statementlist;
        
};

class DeclarationStatement : public ASTNode
{
    public:
        DeclarationStatement(ASTNode *type, ASTNode *varname)
            : type(type), varname(varname){}
        
        DeclarationStatement(){}

        virtual void Accept(Visitor *visitor) const override{
            visitor->visitDeclarationStmt(this);
        }
        
        ~DeclarationStatement(){
            delete type;
            delete varname;
        }

        ASTNode* gettype() const{return type;}
        ASTNode* getvarname() const{return varname;}
    
    private:
        ASTNode *type;
        ASTNode *varname;
};

class ForallStatement : public ASTNode
{
    public:
        ForallStatement(ASTNode *type, ASTNode *varname)
            : type(type), varname(varname){}
        
        ForallStatement(){}

        virtual void Accept(Visitor *visitor) const override{
            visitor->visitForallStmt(this);
        }
        
        ~ForallStatement(){
            delete type;
            delete varname;
        }

        ASTNode* gettype() const{return type;}
        ASTNode* getvarname() const{return varname;}
    
    private:
        ASTNode *type;
        ASTNode *varname;
};

class IfStatement : public ASTNode
{
    public:
        IfStatement(ASTNode *type, ASTNode *varname)
            : type(type), varname(varname){}
        
        IfStatement(){}

        virtual void Accept(Visitor *visitor) const override{
            visitor->visitIfStmt(this);
        }
        
        ~IfStatement(){
            delete type;
            delete varname;
        }

        ASTNode* gettype() const{return type;}
        ASTNode* getvarname() const{return varname;}
    
    private:
        ASTNode *type;
        ASTNode *varname;
};


class BoolExpr : public ASTNode
{
    public:
        BoolExpr(ASTNode *type, ASTNode *varname)
            : type(type), varname(varname){}
        
        BoolExpr(){}

        virtual void Accept(Visitor *visitor) const override{
            visitor->visitBoolExpr(this);
        }
        
        ~BoolExpr(){
            delete type;
            delete varname;
        }

        ASTNode* gettype() const{return type;}
        ASTNode* getvarname() const{return varname;}
    
    private:
        ASTNode *type;
        ASTNode *varname;
};

class Incandassignstmt : public ASTNode
{
    public:
        Incandassignstmt(ASTNode *type, ASTNode *varname)
            : type(type), varname(varname){}
        
        Incandassignstmt(){}

        virtual void Accept(Visitor *visitor) const override{
            visitor->visitIncandassignstmt(this);
        }
        
        ~Incandassignstmt(){
            delete type;
            delete varname;
        }

        ASTNode* gettype() const{return type;}
        ASTNode* getvarname() const{return varname;}
    
    private:
        ASTNode *type;
        ASTNode *varname;
};

class Identifier : public ASTNode
{
    public:
        Identifier(char *name)
            : name_(name){}
        
        Identifier(){}

        virtual void Accept(Visitor *visitor) const override{
            visitor->visitIdentifier(this);
        }
        
        ~Identifier(){
            delete name_;
        }

        char* getname() const{return name_;}
    
    private:
        char* name_;
};

class ReturnStmt : public ASTNode
{
    public:
        ReturnStmt(ASTNode *type, ASTNode *varname)
            : type(type), varname(varname){}
        
        ReturnStmt(){}

        virtual void Accept(Visitor *visitor) const override{
            visitor->visitReturnStmt(this);
        }
        
        ~ReturnStmt(){
            delete type;
            delete varname;
        }

        ASTNode* gettype() const{return type;}
        ASTNode* getvarname() const{return varname;}
    
    private:
        ASTNode *type;
        ASTNode *varname;
};

class Function : public ASTNode
{
    public:
        Function(Identifier *functionname, Arglist *arglist, ASTNode *stmtlist)
            : functionname(functionname), arglist(arglist), stmtlist(stmtlist){

            }
        
        // Function(){}

        virtual void Accept(Visitor *visitor) const override{
            visitor->visitFunction(this);
        }
        
        ~Function(){
            delete functionname;
            delete stmtlist;
        }

        ASTNode* getfuncname() const{return functionname;}
        Arglist* getparams() const{return arglist;}
        ASTNode* getstmtlist() const{return stmtlist;}
    
    private:
        ASTNode* functionname;
        Arglist* arglist;
        ASTNode* stmtlist;
};

class Paramlist : public ASTNode
{
    public:
    
        Paramlist(){}

        virtual void Accept(Visitor *visitor) const override{
            visitor->visitParamlist(this);
        }
        
        ~Paramlist(){
            for(ASTNode* param: paramlist)
                delete param;
        }

        const vector<ASTNode*> getParamList() const{return paramlist;}
        void addparam(ASTNode* param){paramlist.push_back(param);}
    
    private:
        vector<ASTNode*> paramlist;
        
};


class Arglist : public ASTNode
{
    public:
    
        Arglist(){}

        virtual void Accept(Visitor *visitor) const override{
            visitor->visitArglist(this);
        }
        
        ~Arglist(){
            for(ASTNode* arg: arglist)
                delete arg;
        }

        const vector<ASTNode*> getArgList() const{return arglist;}
        void addarg(ASTNode* arg){arglist.push_back(arg);}
    
    private:
        vector<ASTNode*> arglist;
        
};

class Arg : public ASTNode
{
    public:
    
        Arg(){}

        virtual void Accept(Visitor *visitor) const override{
            visitor->visitArg(this);
        }
        
        ~Arg(){
                delete arg;
        }

        const ASTNode* getArg() const{return arg;}
    
    private:
        ASTNode* arg;
        
};

#endif