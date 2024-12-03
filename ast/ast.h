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

class Methodcall : public ASTNode
{
    public:
    
        Methodcall(Identifier* identifier, ASTNode* node)
                    : identifier(identifier), node(node) {}

        Methodcall(Identifier* identifier)
                    : identifier(identifier), node(nullptr){}
        
        virtual void Accept(Visitor *visitor) const override{
            visitor->visitMethodcall(this);
        }
        
        ~Methodcall(){
            delete identifier;
            delete node;
        }

        const Identifier* getIdentifier() const {return identifier;} 
        const ASTNode* getnode() const {return node;}       

    private:
        const Identifier* identifier;
        const ASTNode* node;
        
};

class Memberaccess : public ASTNode
{
    public:
    
        Memberaccess(Identifier* identifier, ASTNode* node)
                    : identifier(identifier), node(node) {}
        


        virtual void Accept(Visitor *visitor) const override{
            visitor->visitMemberaccess(this);
        }
        
        ~Memberaccess(){
            delete identifier;
            delete node;
        }

        const Identifier* getIdentifier()const {return identifier;} 
        const ASTNode* getnode()const{return node;}       

    private:
        const Identifier* identifier;
        const ASTNode* node;
        
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
        
        DeclarationStatement(ASTNode* type, ASTNode* identifier, ASTNode* number) 
            : type(type), varname(identifier), number(number){}

        virtual void Accept(Visitor *visitor) const override{
            visitor->visitDeclarationStmt(this);
        }
        
        ~DeclarationStatement(){
            delete type;
            delete varname;
            delete number;
        }

        ASTNode* gettype() const{return type;}
        ASTNode* getvarname() const{return varname;}
        ASTNode* getnumber() const{return number;}
        
    
    private:
        ASTNode *type;
        ASTNode *varname;
        ASTNode *number;
};

class ForallStatement : public ASTNode
{
    public:
        ForallStatement(Identifier *loopVar, ASTNode *expr, ASTNode* blcstmt)
            : loopVar(loopVar), expr(expr), blcstmt(blcstmt){}
        
        ForallStatement(){}

        virtual void Accept(Visitor *visitor) const override{
            visitor->visitForallStmt(this);
        }
        
        ~ForallStatement(){
            delete loopVar;
            delete expr;
            delete blcstmt;
        }

        Identifier* getLoopVar() const{return loopVar;}
        ASTNode* getexpr() const{return expr;}
        ASTNode* getblcstmt() const{return blcstmt;}
    
    private:
        Identifier *loopVar;
        ASTNode *expr;
        ASTNode *blcstmt;
};

class IfStatement : public ASTNode
{
    public:
        IfStatement(ASTNode *expr, ASTNode *stmt)
            : expr(expr), stmt(stmt){}
        
        IfStatement(){}

        virtual void Accept(Visitor *visitor) const override{
            visitor->visitIfStmt(this);
        }
        
        ~IfStatement(){
            delete expr;
            delete stmt;
        }

        ASTNode* getexpr() const{return expr;}
        ASTNode* getstmt() const{return stmt;}
    
    private:
        ASTNode *expr;
        ASTNode *stmt;
};


class BoolExpr : public ASTNode
{
    public:
        BoolExpr(Identifier *identifier1, char *op,  Identifier *identifier2)
            : identifier1(identifier1), op(op), identifier2(identifier2){}
        
        BoolExpr(){}

        virtual void Accept(Visitor *visitor) const override{
            visitor->visitBoolExpr(this);
        }
        
        ~BoolExpr(){
            delete identifier1;
            delete identifier2;
            delete op;
        }

        Identifier* getIdentifier1() const{return identifier1;}
        Identifier* getIdentifier2() const{return identifier2;}
        char* getop() const{return op;}
    
    private:
        Identifier *identifier1;
        Identifier *identifier2;
        char *op;
};

class Incandassignstmt : public ASTNode
{
    public:
        Incandassignstmt(Identifier *identifier, char *op, ASTNode *expr)
            : identifier(identifier),op(op), expr(expr){}
        
        Incandassignstmt(){}

        virtual void Accept(Visitor *visitor) const override{
            visitor->visitIncandassignstmt(this);
        }
        
        ~Incandassignstmt(){
            delete identifier;
            delete op;
            delete expr;
        }

        Identifier* getIdentifier() const{return identifier;}
        ASTNode* getexpr() const{return expr;}
    
    private:
        Identifier *identifier;
        char *op;
        ASTNode *expr;
};



class Number : public ASTNode
{
    public:
        Number(char *number)
            : number_(atoi(number)){}
        
        Number(){}

        virtual void Accept(Visitor *visitor) const override{
            visitor->visitNumber(this);
        }
        

        int getnumber() const{return number_;}
    
    private:
        int number_;
};

class ReturnStmt : public ASTNode
{
    public:
        ReturnStmt(ASTNode *expr)
            : expr(expr){}
        
        virtual void Accept(Visitor *visitor) const override{
            visitor->visitReturnStmt(this);
        }
        
        ~ReturnStmt(){
            delete expr;
        }

        ASTNode* getexpr() const{return expr;}
    
    private:
        ASTNode *expr;
};

class Function : public ASTNode
{
    public:
        Function(Identifier *functionname, Arglist *arglist, ASTNode *stmtlist)
            : functionname(functionname), arglist(arglist), stmtlist(stmtlist){

            funcName = functionname->getname();

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
        string getfuncNameIdentifier() const{return funcName;} // Returns the function name as a string. 
    
    private:
        ASTNode* functionname;
        Arglist* arglist;
        ASTNode* stmtlist;
        string funcName;
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


class TypeExpr : public ASTNode
{
    public:
    
        TypeExpr(char* type):type_(type){}

        virtual void Accept(Visitor *visitor) const override{
            visitor->visitType(this);
        }
        
        ~TypeExpr(){
                delete type_;
        }

        const char* getType() const{return type_;}
    
    private:
        char* type_;
        
};

class Arg : public ASTNode
{
    public:
    
        Arg(TypeExpr* type, Identifier* identifier) 
            : type(type), varname(identifier){}

        virtual void Accept(Visitor *visitor) const override{
            visitor->visitArg(this);
        }
        
        ~Arg(){
                delete type;
                delete varname;
        }

        const TypeExpr* getType() const{return type;}
        const Identifier* getVarName() const{return varname;}
    
    private:
        TypeExpr* type;
        Identifier* varname;
        
};

class Arglist : public ASTNode
{
    public:
    
        Arglist(){}

        virtual void Accept(Visitor *visitor) const override{
            visitor->visitArglist(this);
        }
        
        ~Arglist(){
            for(Arg* arg: arglist)
                delete arg;
        }

        const vector<Arg*> getArgList() const{return arglist;}
        void addarg(Arg* arg){arglist.push_back(arg);}
    
    private:
        vector<Arg*> arglist;
        
};

class Expression : public ASTNode
{
    public:
    
        Expression(){}

        virtual void Accept(Visitor *visitor) const override{
            visitor->visitExpression(this);
        }

        const ASTNode* getExpression() const{return expr;}
    
    private:
        ASTNode* expr;
        
};



#endif